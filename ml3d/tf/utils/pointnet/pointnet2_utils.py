import tensorflow as tf
from tensorflow.python.framework import ops

import open3d

if open3d.core.cuda.device_count() > 0:
    from open3d.ml.tf.ops import furthest_point_sampling, three_nn, three_interpolate, three_interpolate_grad, ball_query


def furthest_point_sample(xyz, npoint):
    """Uses iterative furthest point sampling to select a set of npoint features
    that have the largest minimum distance.

    :param xyz: (B, N, 3) where N > npoint
    :param npoint: int, number of features in the sampled set
    :return:tensor containing the set
    """
    if not open3d.core.cuda.device_count() > 0:
        raise NotImplementedError

    output = furthest_point_sampling(xyz, npoint)
    return output


ops.NoGradient('Open3DFurthestPointSampling')


def furthest_point_sample_v2(xyz, row_splits, new_row_splits):
    """Furthest Point Sampling with variable length batch support.

    Args:
        xyz (tf.float32): Input pointcloud (N, 3).
        row_splits (tf.int64): splits to define batch (b + 1,)
        new_row_splits (tf.int64): splits for output batch lengths (b + 1,)

    Returns:
        Returns indices of sampled points with shape (new_row_splits[-1], ).
    """
    if not open3d.core.cuda.device_count() > 0:
        raise NotImplementedError

    idx = []
    for i in range(tf.shape(row_splits)[0] - 1):
        npoint = new_row_splits[i + 1] - new_row_splits[i]
        start_i = row_splits[i]
        end_i = row_splits[i + 1]
        out = ml_ops.furthest_point_sampling(
            tf.expand_dims(xyz[start_i:end_i], 0), npoint) + row_splits[i]

        idx.append(out[0])

    return tf.concat(idx, 0)


def three_nn_gpu(query_pts, data_pts):
    """Find the three nearest neighbors of query_pts in data_pts.

    :param query_pts: (B, N, 3)
    :param data_pts: (B, M, 3)
    :return:
        dist: (B, N, 3) l2 distance to the three nearest neighbors
        idx: (B, N, 3) index of 3 nearest neighbors
    """
    if not open3d.core.cuda.device_count() > 0:
        raise NotImplementedError

    dist2, idx = three_nn(query_pts, data_pts)
    return tf.sqrt(dist2), idx


ops.NoGradient("Open3DTreeNN")


def three_interpolate_gpu(features, idx, weight):
    """Performs weight linear interpolation on 3 features.

    :param features: (B, C, M) Features descriptors to be interpolated from
    :param idx: (B, n, 3) three nearest neighbors of the target features in features
    :param weight: (B, n, 3) weights
    :return:
        output: (B, C, N) tensor of the interpolated features
    """
    if not open3d.core.cuda.device_count() > 0:
        raise NotImplementedError

    output = three_interpolate(features, idx, weight)
    return output


@tf.RegisterGradient("Open3DThreeInterpolate")
def _tree_interpolate_gradient(op, grad_out):
    if not open3d.core.cuda.device_count() > 0:
        raise NotImplementedError

    features = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]

    m = features.shape[2]

    grad_features = three_interpolate_grad(grad_out, idx, weight, m)
    return grad_features, None, None


def ball_query_gpu(radius, nsample, xyz, new_xyz):
    """
    :param radius: float, radius of the balls
    :param nsample: int, maximum number of features in the balls
    :param xyz: (B, N, 3) xyz coordinates of the features
    :param new_xyz: (B, npoint, 3) centers of the ball query
    :return:
        idx: (B, npoint, nsample) tensor with the indices of the features that form the query balls
    """
    if not open3d.core.cuda.device_count() > 0:
        raise NotImplementedError

    idx = ball_query(xyz, new_xyz, radius, nsample)
    return idx


ops.NoGradient("Open3DBallQuery")


class QueryAndGroup(tf.keras.layers.Layer):

    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def call(self, xyz, new_xyz, features=None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        idx = ball_query_gpu(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = tf.gather(xyz, idx, batch_dims=1)
        grouped_xyz = tf.transpose(grouped_xyz,
                                   (0, 3, 1, 2))  # (B, 3, npoint, nsample)

        grouped_xyz -= tf.expand_dims(tf.transpose(new_xyz, (0, 2, 1)), axis=-1)

        if features is not None:
            grouped_features = tf.gather(tf.transpose(features, (0, 2, 1)),
                                         idx,
                                         batch_dims=1)
            grouped_features = tf.transpose(
                grouped_features, (0, 3, 1, 2))  # (B, 3, npoint, nsample)
            if self.use_xyz:
                new_features = tf.concat([grouped_xyz, grouped_features],
                                         axis=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(tf.keras.layers.Layer):

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def call(self, xyz, new_xyz, features=None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        grouped_xyz = tf.expand_dims(tf.transpose(xyz, (0, 2, 1)), axis=2)
        if features is not None:
            grouped_features = tf.expand_dims(features, axis=2)
            if self.use_xyz:
                new_features = tf.concat([grouped_xyz, grouped_features],
                                         axis=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
