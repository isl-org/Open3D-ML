import tensorflow as tf
from tensorflow.python.framework import ops
import open3d
if open3d.core.cuda.device_count() > 0:
    from open3d.ml.tf.ops import roi_pool
import numpy as np


def enlarge_box3d(boxes3d, extra_width):
    """Enlarge 3D boxes.

    Args:
        boxes3d: (N, 7) [x, y, z, h, w, l, ry]
        extra_width: Extra width.
    """
    trans = np.zeros((boxes3d.shape[-1],))
    trans[1] = extra_width
    trans[3:6] = extra_width * 2

    return boxes3d + trans


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
    pooled_boxes3d = tf.reshape(
        enlarge_box3d(tf.reshape(boxes3d, (-1, 7)), pool_extra_width),
        (batch_size, -1, 7))

    pooled_features, pooled_empty_flag = roi_pool(pts, pooled_boxes3d,
                                                  pts_feature, sampled_pt_num)

    return pooled_features, pooled_empty_flag


ops.NoGradient('Open3DRoiPool')
