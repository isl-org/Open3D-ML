"""Helpers for Open3D ML PyTorch custom ops (CPU, CUDA, and SYCL/XPU)."""

import open3d


def pytorch_ops_built():
    """Return True if this Open3D build includes PyTorch custom ops."""
    return bool(open3d._build_config.get("BUILD_PYTORCH_OPS"))


def require_pytorch_ops():
    """Raise NotImplementedError when PyTorch ops were not built."""
    if not pytorch_ops_built():
        raise NotImplementedError(
            "Open3D was built without PyTorch ops (BUILD_PYTORCH_OPS=OFF).")
