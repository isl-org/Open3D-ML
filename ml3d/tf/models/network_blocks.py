import tensorflow as tf
import numpy as np

from .utils.kernels.kernel_points import load_kernels as create_kernel_points


def radius_gaussian(sq_r, sig, eps=1e-9):
    """Compute a radius gaussian (gaussian of distance)

    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return tf.exp(-sq_r / (2 * tf.square(sig) + eps))


def max_pool(x, inds):
    """Pools features with the maximum values.

    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """
    # Add a last row with minimum features for shadow pools
    x = tf.concat([x, tf.math.reduce_min(x, axis=0, keepdims=True)],
                  axis=0)  # TODO : different in pytorch.

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = tf.gather(x, inds, axis=0)

    # Pool the maximum [n2, d]
    return tf.reduce_max(pool_features, axis=1)


def closest_pool(x, inds):
    """This tensorflow operation compute a pooling according to the list of
    indices 'inds'.

    > x = [n1, d] features matrix
    > inds = [n2, max_num] We only use the first column of this which should be the closest points too pooled positions
    >> output = [n2, d] pooled features matrix
    """
    # Add a last row with minimum features for shadow pools
    x = tf.concat([x, tf.zeros((1, int(x.shape[1])), x.dtype)], axis=0)

    # Get features for each pooling cell [n2, d]
    pool_features = tf.gather(x, inds[:, 0], axis=0)

    return pool_features


def global_average(x, batch_lengths):
    """Block performing a global average over batch pooling.

    :param x: [N, D] input features
    :param batch_lengths: [B] list of batch lengths
    :return: [B, D] averaged features
    """
    # Loop over the clouds of the batch
    averaged_features = []
    i = 0
    for b_i, length in enumerate(batch_lengths):

        # Average features for each batch cloud
        averaged_features.append(tf.reduce_mean(x[i:i + length], axis=0))

        # Increment for next cloud
        i += length

    # Average features in each batch
    return tf.stack(averaged_features)


def block_decider(block_name, radius, in_dim, out_dim, layer_ind, cfg):

    if block_name == 'unary':
        return UnaryBlock(in_dim,
                          out_dim,
                          cfg.use_batch_norm,
                          cfg.batch_norm_momentum,
                          l_relu=cfg.get('l_relu', 0.2))

    elif block_name in [
            'simple', 'simple_deformable', 'simple_invariant',
            'simple_equivariant', 'simple_strided', 'simple_deformable_strided',
            'simple_invariant_strided', 'simple_equivariant_strided'
    ]:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, cfg)

    elif block_name in [
            'resnetb', 'resnetb_invariant', 'resnetb_equivariant',
            'resnetb_deformable', 'resnetb_strided',
            'resnetb_deformable_strided', 'resnetb_equivariant_strided',
            'resnetb_invariant_strided'
    ]:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius,
                                     layer_ind, cfg)

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)

    elif block_name == 'global_average':
        return GlobalAverageBlock()

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError(
            'Unknown block name in the architecture definition : ' + block_name)


class KPConv(tf.keras.layers.Layer):

    def __init__(self,
                 kernel_size,
                 p_dim,
                 in_channels,
                 out_channels,
                 KP_extent,
                 radius,
                 fixed_kernel_points='center',
                 KP_influence='linear',
                 aggregation_mode='sum',
                 deformable=False,
                 modulated=False,
                 repulse_extent=1.2,
                 deform_fitting_power=1.0,
                 offset_param=False,
                 **kwargs):
        """Initialize parameters for Kernel Point Convolution.

        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius.
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not.
        :param modulated: choose if kernel weights are modulated in addition to deformed.
        """
        super(KPConv, self).__init__(**kwargs)

        self.KP_extent = KP_extent
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        if (offset_param):
            self.wts = self.add_weight(name="{}_W_deform".format(self.name),
                                       shape=(self.K, self.in_channels,
                                              self.out_channels),
                                       initializer='random_normal',
                                       trainable=True)
        else:
            self.wts = self.add_weight(name="{}_W".format(self.name),
                                       shape=(self.K, self.in_channels,
                                              self.out_channels),
                                       initializer='random_normal',
                                       trainable=True)
        self.repulse_extent = repulse_extent
        self.deform_fitting_power = deform_fitting_power

        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(self.K,
                                      self.p_dim,
                                      self.in_channels,
                                      self.offset_dim,
                                      KP_extent,
                                      radius,
                                      fixed_kernel_points=fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode,
                                      offset_param=True)
            self.offset_bias = self.add_weight(name="{}_b".format(self.name),
                                               shape=(self.offset_dim,),
                                               initializer='zeros',
                                               trainable=True)

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        self.reset_parameters()

        if deformable:
            self.kernel_points = self.offset_conv.kernel_points
        else:
            self.kernel_points = self.init_KP()

    def reset_parameters(self):
        return

    def init_KP(self):
        K_points_numpy = create_kernel_points(self.radius,
                                              self.K,
                                              dimension=self.p_dim,
                                              fixed=self.fixed_kernel_points)

        return tf.Variable(K_points_numpy.astype(np.float32),
                           trainable=False,
                           name='kernel_points')

    def regular_loss(self):

        fitting_loss = 0
        repulsive_loss = 0

        if self.deformable:
            KP_min_d2 = self.min_d2 / (self.KP_extent**2)

            fitting_loss += tf.sqrt(tf.nn.l2_loss(KP_min_d2))

            KP_locs = self.deformed_KP / self.KP_extent

            for i in range(self.K):
                other_KP = tf.stop_gradient(
                    tf.concat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]],
                              axis=1))
                distances = tf.sqrt(
                    tf.reduce_sum(tf.square(other_KP - KP_locs[:, i:i + 1, :]),
                                  axis=2))
                rep_loss = tf.reduce_sum(tf.square(
                    tf.maximum(0.0, self.repulse_extent - distances)),
                                         axis=1)
                repulsive_loss += tf.sqrt(tf.nn.l2_loss(rep_loss)) / self.K

        return self.deform_fitting_power * (2 * fitting_loss + repulsive_loss)

    def call(self, query_points, support_points, neighbors_indices, features):

        n_kp = int(self.kernel_points.shape[0])

        if self.deformable:
            # Get offsets with a KPConv that only takes part of the features
            self.offset_features = self.offset_conv(
                query_points, support_points, neighbors_indices,
                features) + self.offset_bias

            if self.modulated:
                # Get offset (in normalized scale) from features
                unscaled_offsets = self.offset_features[:, :self.p_dim * self.K]
                unscaled_offsets = tf.reshape(unscaled_offsets,
                                              (-1, self.K, self.p_dim))

                # Get modulations
                modulations = 2 * tf.sigmoid(
                    self.offset_features[:, self.p_dim * self.K:])

            else:
                # Get offset (in normalized scale) from features
                unscaled_offsets = tf.reshape(self.offset_features,
                                              (-1, self.K, self.p_dim))

                # No modulations
                modulations = None

            # Rescale offset for this layer
            offsets = unscaled_offsets * self.KP_extent

        else:
            offsets = None
            modulations = None

        # Add a fake point in the last row for shadow neighbors
        shadow_point = tf.ones_like(support_points[:1, :]) * 1e6
        support_points = tf.concat([support_points, shadow_point], axis=0)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = tf.gather(support_points, neighbors_indices, axis=0)

        # Center every neighborhood
        neighbors = neighbors - tf.expand_dims(query_points, 1)

        if (self.deformable):
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = tf.expand_dims(self.deformed_KP, 1)
        else:
            deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors = tf.expand_dims(neighbors, 2)
        neighbors = tf.tile(neighbors,
                            [1, 1, n_kp, 1])  # TODO : not in pytorch ?
        differences = neighbors - deformed_K_points

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = tf.reduce_sum(tf.square(differences), axis=3)

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:

            # Save distances for loss
            self.min_d2 = tf.reduce_min(sq_distances, axis=1)

            # Boolean of the neighbors in range of a kernel point [n_points, n_neighbors]
            in_range = tf.cast(
                tf.reduce_any(tf.less(sq_distances, self.KP_extent**2), axis=2),
                tf.int32)

            # New value of max neighbors
            new_max_neighb = tf.reduce_max(tf.reduce_sum(in_range, axis=1))

            # For each row of neighbors, indices of the ones that are in range [n_points, new_max_neighb]
            new_neighb_bool, new_neighb_inds = tf.math.top_k(in_range,
                                                             k=new_max_neighb)

            # Gather new neighbor indices [n_points, new_max_neighb]
            new_neighbors_indices = tf.gather(neighbors_indices,
                                              new_neighb_inds,
                                              batch_dims=-1)

            # Gather new distances to KP [n_points, new_max_neighb, n_kpoints]
            sq_distances = tf.gather(sq_distances,
                                     new_neighb_inds,
                                     batch_dims=1)

            # New shadow neighbors have to point to the last shadow point
            new_neighbors_indices *= new_neighb_bool
            new_neighbors_indices += (
                1 - new_neighb_bool) * int(support_points.shape[0] - 1)

        else:
            new_neighbors_indices = neighbors_indices

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = tf.ones_like(sq_distances)
            all_weights = tf.transpose(all_weights, [0, 2, 1])

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = tf.maximum(1 - tf.sqrt(sq_distances) / self.KP_extent,
                                     0.0)
            all_weights = tf.transpose(all_weights, [0, 2, 1])

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = tf.transpose(all_weights, [0, 2, 1])
        else:
            raise ValueError(
                'Unknown influence function type (cfg.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = tf.argmin(sq_distances,
                                      axis=2,
                                      output_type=tf.int32)
            all_weights *= tf.one_hot(
                neighbors_1nn, self.K, axis=1,
                dtype=tf.float32)  # TODO : transpose in pytorch not here ?

        elif self.aggregation_mode != 'sum':
            raise ValueError(
                "Unknown convolution mode. Should be 'closest' or 'sum'")

        features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighborhood_features = tf.gather(features,
                                          new_neighbors_indices,
                                          axis=0)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = tf.matmul(all_weights, neighborhood_features)

        # Apply modulations
        if self.deformable and self.modulated:
            weighted_features *= tf.expand_dims(modulations, 2)

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = tf.transpose(weighted_features, [1, 0, 2])
        kernel_outputs = tf.matmul(weighted_features, self.wts)

        # Convolution sum to get [n_points, out_fdim]
        output_features = tf.reduce_sum(kernel_outputs, axis=0)

        self.add_loss(self.regular_loss())

        return output_features

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(
            self.radius, self.in_channels, self.out_channels)


class BatchNormBlock(tf.keras.layers.Layer):

    def __init__(self, in_dim, use_bn, bn_momentum):
        """Initialize a batch normalization block. If network does not use batch
        normalization, replace with biases.

        :param in_dim: dimension input features.
        :param use_bn: boolean indicating if we use Batch Norm.
        :param bn_momentum: Batch norm momentum.
        """
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim

        if (self.use_bn):
            self.batch_norm = tf.keras.layers.BatchNormalization(
                epsilon=1e-5, momentum=bn_momentum)
        else:
            self.bias = self.add_weight(name="{}_b".format(self.name),
                                        shape=(self.in_dim,),
                                        initializer='zeros',
                                        trainable=True)

    def call(self, x, training=False):
        if (self.use_bn):
            return self.batch_norm(x, training)
        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(in_feat: {:d}, momentum: {:.3f}, only_bias: {:s})'.format(
            self.in_dim, self.bn_momentum, str(not self.use_bn))


class UnaryBlock(tf.keras.layers.Layer):

    def __init__(self,
                 in_dim,
                 out_dim,
                 use_bn,
                 bn_momentum,
                 no_relu=False,
                 l_relu=0.2):
        """Initialize a standard unary block with its ReLU and BatchNorm.

        :param in_dim: dimension input features.
        :param out_dim: dimension input features.
        :param use_bn: boolean indicating if we use Batch Norm.
        :param bn_momentum: Batch norm momentum.
        """
        super(UnaryBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.mlp = tf.keras.layers.Dense(out_dim, use_bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)

        if not no_relu:
            self.leaky_relu = tf.keras.layers.LeakyReLU(l_relu)

    def call(self, x, batch=None, training=False):
        x = self.mlp(x)
        x = self.batch_norm(x, training)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(
            self.in_dim, self.out_dim, str(self.use_bn), str(not self.no_relu))


class SimpleBlock(tf.keras.layers.Layer):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, cfg):
        """Initialize a simple convolution block with its ReLU and BatchNorm.

        :param in_dim: dimension input features.
        :param out_dim: dimension input features.
        :param radius: current radius of convolution.
        :param cfg: parameters.
        """
        super(SimpleBlock, self).__init__()

        current_extent = radius * cfg.KP_extent / cfg.conv_radius

        self.bn_momentum = cfg.batch_norm_momentum
        self.use_bn = cfg.use_batch_norm
        self.layer_ind = layer_ind
        self.block_name = block_name
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.KPConv = KPConv(cfg.num_kernel_points,
                             cfg.in_points_dim,
                             in_dim,
                             out_dim // 2,
                             current_extent,
                             radius,
                             fixed_kernel_points=cfg.fixed_kernel_points,
                             aggregation_mode=cfg.aggregation_mode,
                             modulated=cfg.modulated)

        self.batch_norm = BatchNormBlock(out_dim // 2, self.use_bn,
                                         self.bn_momentum)
        self.leaky_relu = tf.keras.layers.LeakyReLU(cfg.get('l_relu', 0.2))

    def call(self, x, batch, training=False):

        # TODO : check x, batch
        if 'strided' in self.block_name:
            q_pts = batch['points'][self.layer_ind + 1]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['pools'][self.layer_ind]
        else:
            q_pts = batch['points'][self.layer_ind]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['neighbors'][self.layer_ind]

        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        x = self.batch_norm(x, training)
        x = self.leaky_relu(x)
        return x


class IdentityBlock(tf.keras.layers.Layer):

    def __init__(self):
        """Initialize an Identity block."""
        super(IdentityBlock, self).__init__()

    def call(self, x, training=False):
        return tf.identity(x)


class ResnetBottleneckBlock(tf.keras.layers.Layer):

    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, cfg):
        """Initialize a resnet bottleneck block.

        :param in_dim: dimension input features.
        :param out_dim: dimension input features.
        :param radius: current radius of convolution.
        :param cfg: parameters.
        """
        super(ResnetBottleneckBlock, self).__init__()

        # get KP_extent from current radius
        current_extent = radius * cfg.KP_extent / cfg.conv_radius

        # Get other parameters
        self.bn_momentum = cfg.batch_norm_momentum
        self.use_bn = cfg.use_batch_norm
        self.block_name = block_name
        self.layer_ind = layer_ind
        self.in_dim = in_dim
        self.out_dim = out_dim
        l_relu = cfg.get('l_relu', 0.2)

        # First downscaling mlp
        if in_dim != out_dim // 4:
            self.unary1 = UnaryBlock(in_dim,
                                     out_dim // 4,
                                     self.use_bn,
                                     self.bn_momentum,
                                     l_relu=l_relu)
        else:
            self.unary1 = tf.identity()

        # KPConv block
        self.KPConv = KPConv(cfg.num_kernel_points,
                             cfg.in_points_dim,
                             out_dim // 4,
                             out_dim // 4,
                             current_extent,
                             radius,
                             fixed_kernel_points=cfg.fixed_kernel_points,
                             KP_influence=cfg.KP_influence,
                             aggregation_mode=cfg.aggregation_mode,
                             deformable='deform' in block_name,
                             modulated=cfg.modulated,
                             repulse_extent=cfg.repulse_extent,
                             deform_fitting_power=cfg.deform_fitting_power)

        self.batch_norm_conv = BatchNormBlock(out_dim // 4, self.use_bn,
                                              self.bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(out_dim // 4,
                                 out_dim,
                                 self.use_bn,
                                 self.bn_momentum,
                                 no_relu=True,
                                 l_relu=l_relu)

        # Shortcut optional mpl
        if in_dim != out_dim:
            self.unary_shortcut = UnaryBlock(in_dim,
                                             out_dim,
                                             self.use_bn,
                                             self.bn_momentum,
                                             no_relu=True,
                                             l_relu=l_relu)
        else:
            self.unary_shortcut = IdentityBlock()

        # Other operations
        self.leaky_relu = tf.keras.layers.LeakyReLU(l_relu)

        return

    def call(self, features, batch, training=False):

        if 'strided' in self.block_name:
            q_pts = batch['points'][self.layer_ind + 1]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['pools'][self.layer_ind]
        else:
            q_pts = batch['points'][self.layer_ind]
            s_pts = batch['points'][self.layer_ind]
            neighb_inds = batch['neighbors'][self.layer_ind]

        # First downscaling mlp
        x = self.unary1(features, training)

        # Convolution
        x = self.KPConv(q_pts, s_pts, neighb_inds, x)
        x = self.leaky_relu(self.batch_norm_conv(x, training))

        # Second upscaling mlp
        x = self.unary2(x, training)

        # Shortcut
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighb_inds)  # TODO : test max_pool
        else:
            shortcut = features
        shortcut = self.unary_shortcut(shortcut, training)

        return self.leaky_relu(x + shortcut)

    def __repr__(self):
        return 'ResnetBlock(in_feat: {:d}, out_feat: {:d})'.format(
            self.in_dim, self.out_dim)


class NearestUpsampleBlock(tf.keras.layers.Layer):

    def __init__(self, layer_ind):
        """Initialize a nearest upsampling block."""
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def call(self, x, batch):
        return closest_pool(x, batch['upsamples'][self.layer_ind - 1])

    def __repr__(self):
        return 'NearestUpsampleBlock(layer: {:d} -> {:d})'.format(
            self.layer_ind, self.layer_ind - 1)


class MaxPoolBlock(tf.keras.layers.Layer):

    def __init__(self, layer_ind):
        """Initialize a Max Pool block."""
        super(MaxPoolBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return max_pool(x, batch['pools'][self.layer_ind + 1])


class GlobalAverageBlock(tf.keras.layers.Layer):

    def __init__(self):
        """Initialize a global average block."""
        super(GlobalAverageBlock, self).__init__()
        return

    def forward(self, x, batch):
        return global_average(x, batch.lengths[-1])
