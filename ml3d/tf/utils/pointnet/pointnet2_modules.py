import tensorflow as tf

from . import pointnet2_utils
from . import tf_utils
from typing import List


class _PointnetSAModuleBase(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def call(self, xyz, features=None, new_xyz=None, training=True):
        r"""
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        if new_xyz is None and self.npoint is not None:
            sampling = tf.expand_dims(pointnet2_utils.furthest_point_sample(
                xyz, self.npoint),
                                      axis=-1)
            new_xyz = tf.gather_nd(xyz, sampling, batch_dims=1)

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz,
                                            features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](
                new_features,
                training=training)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = tf.nn.max_pool2d(new_features,
                                                kernel_size=[
                                                    1, new_features.size(3)
                                                ])  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = tf.nn.avg_pool2d(new_features,
                                                kernel_size=[
                                                    1, new_features.size(3)
                                                ])  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = tf.squeeze(new_features,
                                      axis=-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, tf.concat(new_features_list, axis=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping."""

    def __init__(self,
                 *,
                 npoint: int,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = []
        self.mlps = []
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(tf_utils.SharedMLP(mlp_spec, bn=bn))
        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer."""

    def __init__(self,
                 *,
                 mlp: List[int],
                 npoint: int = None,
                 radius: float = None,
                 nsample: int = None,
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(mlps=[mlp],
                         npoint=npoint,
                         radii=[radius],
                         nsamples=[nsample],
                         bn=bn,
                         use_xyz=use_xyz,
                         pool_method=pool_method)


class PointnetFPModule(tf.keras.layers.Layer):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.mlp = tf_utils.SharedMLP(mlp, bn=bn)

    def call(self, unknown, known, unknow_feats, known_feats, training=True):
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn_gpu(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = tf.sum(dist_recip, axis=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate_gpu(
                known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2],
                                                    unknown.size(1))

        if unknow_feats is not None:
            new_features = tf.concat([interpolated_feats, unknow_feats],
                                     axis=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = tf.expand_dims(new_features, axis=-1)
        new_features = self.mlp(new_features, training=training)

        return tf.squeeze(new_features, axis=-1)


if __name__ == "__main__":
    pass
