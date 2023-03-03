import tensorflow as tf

from typing import List

from ..utils.tf_utils import gen_CNN

from ...utils import MODEL
from ..utils.pointnet import pointnet2_utils


class Pointnet2MSG(tf.keras.layers.Layer):

    def __init__(
            self,
            in_channels=6,
            use_xyz=True,
            SA_config={
                "npoints": [128, 32, -1],
                "radius": [0.2, 0.4, 100],
                "nsample": [64, 64, 64],
                "mlps": [[128, 128, 128], [128, 128, 256], [256, 256, 512]]
            },
            fp_mlps=[]):
        super().__init__()

        self.SA_modules = []
        skip_channel_list = [in_channels]
        for i in range(len(SA_config["npoints"])):
            mlps = SA_config["mlps"][i].copy()
            out_channels = 0
            for idx in range(len(mlps)):
                mlps[idx] = [in_channels] + mlps[idx]
                out_channels += mlps[idx][-1]
            self.SA_modules.append(
                PointnetSAModuleMSG(npoint=SA_config["npoints"][i],
                                    radii=SA_config["radius"][i],
                                    nsamples=SA_config["nsample"][i],
                                    mlps=mlps,
                                    use_xyz=use_xyz,
                                    batch_norm=True))
            in_channels = out_channels
            skip_channel_list.append(out_channels)

        self.FP_modules = []

        for i in range(len(fp_mlps)):
            pre_channel = fp_mlps[
                i + 1][-1] if i + 1 < len(fp_mlps) else out_channels
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[i]] +
                                 fp_mlps[i],
                                 batch_norm=True))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3]
        features = pc[..., 3:] if pc.shape[-1] > 3 else None

        return xyz, features

    def call(self, pointcloud, training=True):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i],
                                                     l_features[i],
                                                     training=training)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1],
                                                   l_xyz[i],
                                                   l_features[i - 1],
                                                   l_features[i],
                                                   training=training)

        return l_xyz[0], l_features[0]


MODEL._register_module(Pointnet2MSG, 'tf')


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
                new_features = tf.reduce_max(new_features,
                                             axis=-1)  # (B, mlp[-1], npoint)
            elif self.pool_method == 'avg_pool':
                new_features = tf.reduce_mean(new_features,
                                              axis=-1)  # (B, mlp[-1], npoint)
            else:
                raise NotImplementedError

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
                 batch_norm=False,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 use_bias=False):
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

            self.mlps.append(
                gen_CNN(mlp_spec,
                        conv=tf.keras.layers.Conv2D,
                        batch_norm=batch_norm,
                        use_bias=use_bias))

        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer."""

    def __init__(self,
                 *,
                 mlp: List[int],
                 npoint: int = None,
                 radius: float = None,
                 nsample: int = None,
                 batch_norm=False,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 use_bias=False):
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
                         batch_norm=batch_norm,
                         use_xyz=use_xyz,
                         pool_method=pool_method,
                         use_bias=use_bias)


MODEL._register_module(PointnetSAModule, 'tf')


class PointnetFPModule(tf.keras.layers.Layer):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], batch_norm=False, use_bias=False):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.mlp = gen_CNN(mlp,
                           conv=tf.keras.layers.Conv2D,
                           batch_norm=batch_norm,
                           use_bias=use_bias)

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
            norm = tf.reduce_sum(dist_recip, axis=2, keepdims=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate_gpu(
                known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.shape[0:2],
                                                    unknown.shape[1])

        if unknow_feats is not None:
            new_features = tf.concat([interpolated_feats, unknow_feats],
                                     axis=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = tf.expand_dims(new_features, axis=-1)
        new_features = self.mlp(new_features, training=training)

        return tf.squeeze(new_features, axis=-1)


MODEL._register_module(PointnetFPModule, 'tf')

if __name__ == "__main__":
    pass
