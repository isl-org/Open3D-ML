import open3d

if open3d.core.cuda.device_count() > 0:
    # Open3D is built with CUDA and the machine has a CUDA device.
    from open3d.ml.contrib import iou_bev_cuda as iou_bev
    from open3d.ml.contrib import iou_3d_cuda as iou_3d
else:
    from open3d.ml.contrib import iou_bev_cpu as iou_bev
    from open3d.ml.contrib import iou_3d_cpu as iou_3d

from .mAP import precision_3d, mAP

__all__ = ['precision_3d', 'mAP', 'iou_bev', 'iou_3d']
