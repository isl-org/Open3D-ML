import torch
import open3d
if open3d.core.cuda.device_count() > 0:
    from open3d.ml.torch.ops import roi_pool
import numpy as np


def enlarge_box3d(boxes3d, extra_width):
    """Enlarge 3D box.

    Args:
        boxes3d: (N, 7) [x, y, z, h, w, l, ry]
        extra_width: extra width
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 1] += extra_width
    return large_boxes3d


def roipool3d_gpu(pts,
                  pts_feature,
                  boxes3d,
                  pool_extra_width,
                  sampled_pt_num=512):
    """Roipool3D GPU.

    Args:
        pts: (B, N, 3)
        pts_feature: (B, N, C)
        boxes3d: (B, M, 7)
        pool_extra_width: float
        sampled_pt_num: int

    Returns:
        pooled_features: (B, M, 512, 3 + C)
        pooled_empty_flag: (B, M)
    """
    if not open3d.core.cuda.device_count() > 0:
        raise NotImplementedError

    batch_size = pts.shape[0]
    pooled_boxes3d = enlarge_box3d(boxes3d.view(-1, 7),
                                   pool_extra_width).view(batch_size, -1, 7)

    pooled_features, pooled_empty_flag = roi_pool(pts.contiguous(),
                                                  pooled_boxes3d.contiguous(),
                                                  pts_feature.contiguous(),
                                                  sampled_pt_num)

    return pooled_features, pooled_empty_flag
