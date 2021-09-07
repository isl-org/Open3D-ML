import tensorflow as tf
from tensorflow.python.framework import ops

import open3d

if open3d.core.cuda.device_count() > 0:
    import open3d.ml.tf.ops as ml_ops


def trilinear_devoxelize(features, coords, resolution, is_training=True):
    if not open3d.core.cuda.device_count() > 0:
        raise NotImplementedError

    outs, inds, wgts = ml_ops.trilinear_devoxelize(
        tf.transpose(coords, perm=[0, 2, 1]),
        tf.transpose(features, perm=[0, 4, 1, 2, 3]), resolution, is_training)
    return tf.transpose(outs, perm=[0, 2, 1])


@tf.RegisterGradient("Open3DTrilinearDevoxelize")
def _trilinear_devoxelize_gradient(op, grad_out, grad_inds, grad_wgts):
    if not open3d.core.cuda.device_count() > 0:
        raise NotImplementedError

    inds = op.outputs[1]
    wgts = op.outputs[2]
    r = op.attrs[1]

    grad_input = ml_ops.trilinear_devoxelize_grad(grad_out, inds, wgts, r)

    return None, grad_input
