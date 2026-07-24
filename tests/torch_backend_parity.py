"""CPU vs CUDA / XPU parity helpers for Open3D-ML PyTorch model tests."""

import copy

import numpy as np
import open3d as o3d
import pytest
import torch


def parity_accelerator():
    """Torch device for accelerator parity, or None if only CPU applies."""
    cfg = o3d._build_config
    if cfg.get("BUILD_CUDA_MODULE") and torch.cuda.is_available():
        cuda_version = cfg.get("CUDA_VERSION")
        if cuda_version and torch.version.cuda == cuda_version:
            return torch.device("cuda")
    if (cfg.get("BUILD_SYCL_MODULE") and hasattr(torch, "xpu") and
            torch.xpu.is_available()):
        return torch.device("xpu")
    return None


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


def _compare_tensors(ref, actual, rtol, atol, path):
    ref_np = ref.detach().cpu().numpy() if torch.is_tensor(ref) else ref
    act_np = actual.detach().cpu().numpy() if torch.is_tensor(actual) else actual
    if ref_np.shape != act_np.shape:
        pytest.fail(f"Shape mismatch at {path}: {ref_np.shape} vs {act_np.shape}")
    if not np.all(np.isfinite(ref_np)) or not np.all(np.isfinite(act_np)):
        pytest.fail(f"Non-finite values at {path}")
    np.testing.assert_allclose(ref_np, act_np, rtol=rtol, atol=atol,
                               err_msg=f"CPU vs accelerator mismatch at {path}")


def assert_outputs_allclose(ref, actual, rtol=1e-4, atol=1e-5, path="root"):
    """Compare nested model outputs (tensors, lists, tuples, dicts)."""
    if torch.is_tensor(ref) or isinstance(ref, np.ndarray):
        _compare_tensors(ref, actual, rtol, atol, path)
        return
    if isinstance(ref, dict):
        if ref.keys() != actual.keys():
            pytest.fail(f"Dict keys differ at {path}: {ref.keys()} vs {actual.keys()}")
        for key in ref:
            assert_outputs_allclose(ref[key], actual[key], rtol, atol,
                                    f"{path}.{key}")
        return
    if isinstance(ref, (list, tuple)):
        if len(ref) != len(actual):
            pytest.fail(
                f"Length mismatch at {path}: {len(ref)} vs {len(actual)}")
        for idx, (r, a) in enumerate(zip(ref, actual)):
            assert_outputs_allclose(r, a, rtol, atol, f"{path}[{idx}]")
        return
    if ref == actual:
        return
    pytest.fail(f"Unsupported or unequal values at {path}: {type(ref)}")


def assert_cpu_accelerator_parity(cpu_output,
                                  forward_on_device,
                                  rtol=1e-4,
                                  atol=1e-5):
    """Compare in-process CPU forward with ``forward_on_device(accel_device)``.

    Uses the same loaded ``open3d_torch_ops.so``; CPU and accelerator paths
    dispatch inside the op library.
    """
    accel = parity_accelerator()
    if accel is None:
        return

    accel_output = forward_on_device(accel)
    assert_outputs_allclose(cpu_output, accel_output, rtol=rtol, atol=atol)


def clone_module_to_device(module, device):
    """Copy ``module`` weights onto ``device`` for a second forward pass."""
    clone = copy.deepcopy(module)
    dev = device if isinstance(device, torch.device) else torch.device(device)
    clone.to(dev)
    if hasattr(clone, "device"):
        clone.device = str(dev)
    clone.eval()
    return clone
