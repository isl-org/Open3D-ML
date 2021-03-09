import torch
import open3d
if open3d.core.cuda.device_count() > 0:
    from open3d.ml.torch.ops import roipool3d
import numpy as np


def enlarge_box3d(boxes3d, extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
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
    """
    :param pts: (B, N, 3)
    :param pts_feature: (B, N, C)
    :param boxes3d: (B, M, 7)
    :param pool_extra_width: float
    :param sampled_pt_num: int
    :return:
        pooled_features: (B, M, 512, 3 + C)
        pooled_empty_flag: (B, M)
    """
    if not open3d.core.cuda.device_count() > 0:
        raise NotImplementedError

    batch_size = pts.shape[0]
    pooled_boxes3d = enlarge_box3d(boxes3d.view(-1, 7),
                                   pool_extra_width).view(batch_size, -1, 7)

    pooled_features, pooled_empty_flag = roipool3d(pts.contiguous(),
                                                   pooled_boxes3d.contiguous(),
                                                   pts_feature.contiguous(),
                                                   sampled_pt_num)

    return pooled_features, pooled_empty_flag
