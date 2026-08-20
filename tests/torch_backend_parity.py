"""CPU vs CUDA / XPU parity helpers for Open3D-ML PyTorch model tests."""

import open3d as o3d
import torch

# Tolerances for CPU vs accelerator comparisons, keyed by accelerator type.
RTOL = {"cuda": 1e-4, "xpu": 1e-3}
ATOL = {"cuda": 1e-5, "xpu": 1e-4}


def move_to_device(obj, device):
    """Recursively move tensors in nested batch structures to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: move_to_device(val, device) for key, val in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(val, device) for val in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(val, device) for val in obj)
    return obj


def collect_grads(module):
    """Return {param_name: grad} for the parameters that received a gradient."""
    return {
        name: param.grad
        for name, param in module.named_parameters()
        if param.grad is not None
    }


def _synchronize_accel(accel_type):
    if hasattr(torch, "accelerator"):
        torch.accelerator.synchronize()
    elif accel_type == "cuda":
        torch.cuda.synchronize()
    elif accel_type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()


def assert_cpu_accelerator_parity(run_on_device, atol=None, grad_atol=None):
    """Run one forward + backward pass per device and compare outputs and grads.

    ``run_on_device(device)`` builds the model and batch on ``device``, runs
    forward and ``loss.backward()``, and returns ``(output, model)``. Returns the
    CPU ``(output, model)``; the accelerator half is skipped if none is present.
    """
    cpu_output, cpu_model = run_on_device(torch.device("cpu"))

    if o3d.core.cuda.is_available() and torch.cuda.is_available():
        accel = torch.device('cuda')
    elif o3d.core.sycl.is_available() and torch.xpu.is_available():
        # o3d.core.sycl.is_available() is True whenever Open3D's SYCL module was
        # built, even on hosts with no Intel GPU (it falls back to a SYCL CPU
        # device). torch.xpu.is_available() additionally confirms a real XPU
        # device is visible to PyTorch, so this only picks 'xpu' on actual GPU
        # hardware.
        accel = torch.device('xpu')
    else:
        return cpu_output, cpu_model

    accel_output, accel_model = run_on_device(accel)
    _synchronize_accel(accel.type)

    rtol = RTOL[accel.type]
    atol = ATOL[accel.type] if atol is None else atol
    # assert_close walks tensors, dicts and sequences alike. The tensors being
    # compared live on different devices, so only their values must match.
    torch.testing.assert_close(
        accel_output,
        cpu_output,
        rtol=rtol,
        atol=atol,
        check_device=False,
        msg=lambda m: f"{accel.type} vs CPU output mismatch\n{m}")
    torch.testing.assert_close(
        collect_grads(accel_model),
        collect_grads(cpu_model),
        rtol=rtol,
        atol=atol if grad_atol is None else grad_atol,
        check_device=False,
        msg=lambda m: f"{accel.type} vs CPU gradient mismatch\n{m}")
    return cpu_output, cpu_model
