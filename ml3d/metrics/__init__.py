import open3d


def _sycl_accelerator_available():
    """True when Open3D sees a non-CPU SYCL device (e.g. Intel GPU), not SYCL-on-CPU."""
    o3c = open3d.core
    if not o3c.sycl.is_available():
        return False
    for device in o3c.sycl.get_available_devices():
        props = o3c.sycl.get_device_properties(device)
        if props.device_type != "cpu":
            return True
    return False


if open3d.core.cuda.device_count() > 0:
    # Open3D is built with CUDA and the machine has a CUDA device.
    from open3d.ml.contrib import iou_bev_cuda as iou_bev
    from open3d.ml.contrib import iou_3d_cuda as iou_3d
elif _sycl_accelerator_available():
    # Open3D SYCL build with an accelerator-class SYCL device (not CPU-only SYCL).
    from open3d.ml.contrib import iou_bev_sycl as iou_bev
    from open3d.ml.contrib import iou_3d_sycl as iou_3d
else:
    from open3d.ml.contrib import iou_bev_cpu as iou_bev
    from open3d.ml.contrib import iou_3d_cpu as iou_3d

from .mAP import precision_3d, mAP

__all__ = ['precision_3d', 'mAP', 'iou_bev', 'iou_3d']
