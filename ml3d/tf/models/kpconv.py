import time
import sys
import tensorflow as tf
import numpy as np
import random

from os.path import exists, join, isfile, dirname, abspath, split
from os import makedirs
from pathlib import Path
from sklearn.neighbors import KDTree
from tqdm import tqdm

# use relative import for being compatible with Open3d main repo
from open3d.ml.tf.ops import *
from open3d.ml.tf.layers import FixedRadiusSearch
from .network_blocks import *
from .base_model import BaseModel
from ...utils import MODEL
from ...datasets.utils import DataProcessing, trans_normalize
from .network_blocks import *
from open3d.ml.tf.ops import batch_grid_subsampling as tf_batch_subsampling


class KPFCNN(BaseModel):
    """Class defining KPFCNN.

    A model for Semantic Segmentation.
    """

    def __init__(
            self,
            name='KPFCNN',
            lbl_values=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19
            ],
            num_classes=19,  # Number of valid classes
            ignored_label_inds=[0],
            ckpt_path=None,
            batcher='ConcatBatcher',
            architecture=[
                'simple', 'resnetb', 'resnetb_strided', 'resnetb', 'resnetb',
                'resnetb_strided', 'resnetb', 'resnetb', 'resnetb_strided',
                'resnetb', 'resnetb', 'resnetb_strided', 'resnetb',
                'nearest_upsample', 'unary', 'nearest_upsample', 'unary',
                'nearest_upsample', 'unary', 'nearest_upsample', 'unary'
            ],
            in_radius=4.0,
            max_in_points=100000,
            batch_num=8,
            batch_limit=30000,
            val_batch_num=8,
            num_kernel_points=15,
            first_subsampling_dl=0.06,
            conv_radius=2.5,
            deform_radius=6.0,
            KP_extent=1.2,
            KP_influence='linear',
            aggregation_mode='sum',
            density_parameter=5.0,
            first_features_dim=128,
            in_features_dim=2,
            modulated=False,
            use_batch_norm=True,
            batch_norm_momentum=0.02,
            deform_fitting_mode='point2point',
            deform_fitting_power=1.0,
            repulse_extent=1.2,
            augment_scale_anisotropic=True,
            augment_symmetries=[True, False, False],
            augment_rotation='vertical',
            augment_scale_min=0.8,
            augment_scale_max=1.2,
            augment_noise=0.001,
            augment_color=0.8,
            in_points_dim=3,
            fixed_kernel_points='center',
            num_layers=5,
            l_relu=0.1,
            reduce_fc=False,
            **kwargs):

        super().__init__(name=name,
                         lbl_values=lbl_values,
                         num_classes=num_classes,
                         ignored_label_inds=ignored_label_inds,
                         ckpt_path=ckpt_path,
                         batcher=batcher,
                         architecture=architecture,
                         in_radius=in_radius,
                         max_in_points=max_in_points,
                         batch_num=batch_num,
                         batch_limit=batch_limit,
                         val_batch_num=val_batch_num,
                         num_kernel_points=num_kernel_points,
                         first_subsampling_dl=first_subsampling_dl,
                         conv_radius=conv_radius,
                         deform_radius=deform_radius,
                         KP_extent=KP_extent,
                         KP_influence=KP_influence,
                         aggregation_mode=aggregation_mode,
                         density_parameter=density_parameter,
                         first_features_dim=first_features_dim,
                         in_features_dim=in_features_dim,
                         modulated=modulated,
                         use_batch_norm=use_batch_norm,
                         batch_norm_momentum=batch_norm_momentum,
                         deform_fitting_mode=deform_fitting_mode,
                         deform_fitting_power=deform_fitting_power,
                         repulse_extent=repulse_extent,
                         augment_scale_anisotropic=augment_scale_anisotropic,
                         augment_symmetries=augment_symmetries,
                         augment_rotation=augment_rotation,
                         augment_scale_min=augment_scale_min,
                         augment_scale_max=augment_scale_max,
                         augment_noise=augment_noise,
                         augment_color=augment_color,
                         in_points_dim=in_points_dim,
                         fixed_kernel_points=fixed_kernel_points,
                         num_layers=num_layers,
                         l_relu=l_relu,
                         reduce_fc=reduce_fc,
                         **kwargs)

        cfg = self.cfg

        # From config parameter, compute higher bound of neighbors number in a neighborhood
        hist_n = int(np.ceil(4 / 3 * np.pi * (cfg.density_parameter + 1)**3))

        # Initiate neighbors limit with higher bound
        self.neighborhood_limits = []
        # self.neighborhood_limits = np.full(cfg.num_layers,
        #                                    hist_n,
        #                                    dtype=np.int32)

        lbl_values = cfg.lbl_values
        ign_lbls = cfg.ignored_label_inds

        # Current radius of convolution and feature dimension
        layer = 0
        r = cfg.first_subsampling_dl * cfg.conv_radius
        in_dim = cfg.in_features_dim
        out_dim = cfg.first_features_dim
        self.K = cfg.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        # Save all block operations in a list of modules
        self.encoder_blocks = []
        self.encoder_skip_dims = []
        self.encoder_skips = []

        for block_i, block in enumerate(cfg.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError(
                    'Equivariant block but features dimension is not a factor of 3'
                )

            # Detect change to next layer for skip connection
            if np.any([
                    tmp in block
                    for tmp in ['pool', 'strided', 'upsample', 'global']
            ]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(
                block_decider(block, r, in_dim, out_dim, layer, cfg))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        # Decoder blocks
        self.decoder_blocks = []
        self.decoder_concats = []

        if cfg.reduce_fc:
            out_dim = out_dim // 2
        start_i = 0
        for block_i, block in enumerate(cfg.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(cfg.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in cfg.architecture[start_i +
                                                              block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(
                block_decider(block, r, in_dim, out_dim, layer, cfg))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        if reduce_fc:
            self.head_mlp = UnaryBlock(out_dim,
                                       cfg.first_features_dim // 2,
                                       True,
                                       cfg.batch_norm_momentum,
                                       l_relu=cfg.get('l_relu', 0.1))
            self.head_softmax = UnaryBlock(cfg.first_features_dim // 2,
                                           self.C,
                                           False,
                                           0,
                                           no_relu=True,
                                           l_relu=cfg.get('l_relu', 0.1))
        else:
            self.head_mlp = UnaryBlock(out_dim,
                                       cfg.first_features_dim,
                                       False,
                                       0,
                                       l_relu=cfg.get('l_relu', 0.1))
            self.head_softmax = UnaryBlock(cfg.first_features_dim,
                                           self.C,
                                           False,
                                           0,
                                           l_relu=cfg.get('l_relu', 0.1))

        self.valid_labels = np.sort(
            [c for c in lbl_values if c not in ign_lbls])

        return

    def organise_inputs(self, flat_inputs):
        cfg = self.cfg

        inputs = dict()
        inputs['points'] = flat_inputs[:cfg.num_layers]
        inputs['neighbors'] = flat_inputs[cfg.num_layers:2 * cfg.num_layers]
        inputs['pools'] = flat_inputs[2 * cfg.num_layers:3 * cfg.num_layers]
        inputs['upsamples'] = flat_inputs[3 * cfg.num_layers:4 * cfg.num_layers]

        ind = 4 * cfg.num_layers
        inputs['features'] = flat_inputs[ind]
        ind += 1
        inputs['batch_weights'] = flat_inputs[ind]
        ind += 1
        inputs['in_batches'] = flat_inputs[ind]
        ind += 1
        inputs['out_batches'] = flat_inputs[ind]
        ind += 1
        inputs['point_labels'] = flat_inputs[ind]
        ind += 1
        labels = inputs['point_labels']

        inputs['augment_scales'] = flat_inputs[ind]
        ind += 1
        inputs['augment_rotations'] = flat_inputs[ind]

        ind += 1
        inputs['point_inds'] = flat_inputs[ind]
        ind += 1
        inputs['cloud_inds'] = flat_inputs[ind]

        return inputs

    def call(self, flat_inputs, training=False):
        cfg = self.cfg
        inputs = self.organise_inputs(flat_inputs)

        x = tf.stop_gradient(tf.identity(inputs['features']))

        skip_conn = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_conn.append(x)
            x = block_op(x, inputs, training=training)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = tf.concat([x, skip_conn.pop()], axis=1)
            x = block_op(x, inputs, training=training)

        x = self.head_mlp(x, inputs, training)
        x = self.head_softmax(x, inputs, training)

        return x

    def get_optimizer(self, cfg_pipeline):

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=cfg_pipeline.learning_rate,
            momentum=cfg_pipeline.momentum)

        return optimizer

    def get_loss(self, Loss, logits, inputs):
        """Runs the loss on outputs of the model.

        :param outputs: logits
        :param labels: labels
        :return: loss
        """
        cfg = self.cfg
        labels = self.organise_inputs(inputs)['point_labels']

        scores, labels = Loss.filter_valid_label(logits, labels)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)
        loss += sum(self.losses)

        return loss, labels, scores

    def big_neighborhood_filter(self, neighbors, layer):
        """Filter neighborhoods with max number of neighbors.

        Limit is set to keep XX% of the neighborhoods untouched. Limit is
        computed at initialization
        """
        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors

    def get_batch_inds(self, stacks_len):
        """Method computing the batch indices of all points, given the batch
        element sizes (stack lengths).

        Example:
        From [3, 2, 5], it would return [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
        """
        # Initiate batch inds tensor
        num_batches = tf.shape(stacks_len)[0]
        num_points = tf.reduce_sum(stacks_len)
        batch_inds_0 = tf.zeros((num_points,), dtype=tf.int32)

        # Define body of the while loop
        def body(batch_i, point_i, b_inds):

            num_in = stacks_len[batch_i]
            num_before = tf.cond(tf.less(batch_i, 1), lambda: tf.zeros(
                (), dtype=tf.int32),
                                 lambda: tf.reduce_sum(stacks_len[:batch_i]))
            num_after = tf.cond(tf.less(batch_i, num_batches - 1),
                                lambda: tf.reduce_sum(stacks_len[batch_i + 1:]),
                                lambda: tf.zeros((), dtype=tf.int32))

            # Update current element indices
            inds_before = tf.zeros((num_before,), dtype=tf.int32)
            inds_in = tf.fill((num_in,), batch_i)
            inds_after = tf.zeros((num_after,), dtype=tf.int32)
            n_inds = tf.concat([inds_before, inds_in, inds_after], axis=0)

            b_inds += n_inds

            # Update indices
            point_i += stacks_len[batch_i]
            batch_i += 1

            return batch_i, point_i, b_inds

        def cond(batch_i, point_i, b_inds):
            return tf.less(batch_i, tf.shape(stacks_len)[0])

        _, _, batch_inds = tf.while_loop(cond,
                                         body,
                                         loop_vars=[0, 0, batch_inds_0],
                                         shape_invariants=[
                                             tf.TensorShape([]),
                                             tf.TensorShape([]),
                                             tf.TensorShape([None])
                                         ])

        return batch_inds

    def stack_batch_inds(self, stacks_len):

        # Initiate batch inds tensor
        num_points = tf.reduce_sum(stacks_len)
        max_points = tf.reduce_max(stacks_len)
        batch_inds_0 = tf.zeros((0, max_points), dtype=tf.int32)

        # Define body of the while loop
        def body(batch_i, point_i, b_inds):

            # Create this element indices
            element_inds = tf.expand_dims(tf.range(
                point_i, point_i + stacks_len[batch_i]),
                                          axis=0)

            # Pad to right size
            padded_inds = tf.pad(
                element_inds, [[0, 0], [0, max_points - stacks_len[batch_i]]],
                "CONSTANT",
                constant_values=num_points)

            # Concatenate batch indices
            b_inds = tf.concat((b_inds, padded_inds), axis=0)

            # Update indices
            point_i += stacks_len[batch_i]
            batch_i += 1

            return batch_i, point_i, b_inds

        def cond(batch_i, point_i, b_inds):
            return tf.less(batch_i, tf.shape(stacks_len)[0])

        fixed_shapes = [
            tf.TensorShape([]),
            tf.TensorShape([]),
            tf.TensorShape([None, None])
        ]
        _, _, batch_inds = tf.while_loop(cond,
                                         body,
                                         loop_vars=[0, 0, batch_inds_0],
                                         shape_invariants=fixed_shapes)

        # Add a last column with shadow neighbor if there is not
        def f1():
            return tf.pad(batch_inds, [[0, 0], [0, 1]],
                          "CONSTANT",
                          constant_values=num_points)

        def f2():
            return batch_inds

        batch_inds = tf.cond(tf.equal(num_points,
                                      max_points * tf.shape(stacks_len)[0]),
                             true_fn=f1,
                             false_fn=f2)

        return batch_inds

    def augment_input(self, stacked_points, batch_inds, is_test):

        cfg = self.cfg
        # Parameter
        num_batches = batch_inds[-1] + 1

        # Rotation
        if cfg.augment_rotation == 'vertical':

            # Choose a random angle for each element
            theta = tf.random.uniform((num_batches,),
                                      minval=0,
                                      maxval=2 * np.pi)

            # Rotation matrices
            c, s = tf.cos(theta), tf.sin(theta)
            cs0 = tf.zeros_like(c)
            cs1 = tf.ones_like(c)
            R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
            R = tf.reshape(R, (-1, 3, 3))

            # Create N x 3 x 3 rotation matrices to multiply with stacked_points
            stacked_rots = tf.gather(R, batch_inds)

            # Apply rotations
            if not is_test:
                stacked_points = tf.reshape(
                    tf.matmul(tf.expand_dims(stacked_points, axis=1),
                              stacked_rots), [-1, 3])

        elif cfg.augment_rotation == 'none':
            R = tf.eye(3, batch_shape=(num_batches,))

        else:
            raise ValueError('Unknown rotation augmentation : ' +
                             cfg.augment_rotation)

        # Scale
        # Choose random scales for each example
        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max

        if cfg.augment_scale_anisotropic:
            s = tf.random.uniform((num_batches, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random.uniform((num_batches, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(
                    tf.round(tf.random.uniform((num_batches, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([num_batches, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)

        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.gather(s, batch_inds)

        # Apply scales and Noise
        if not is_test:
            stacked_points = stacked_points * stacked_scales

            noise = tf.random.normal(tf.shape(stacked_points),
                                     stddev=cfg.augment_noise)
            stacked_points = stacked_points + noise

        return stacked_points, s, R

    def segmentation_inputs(self,
                            stacked_points,
                            stacked_features,
                            point_labels,
                            stacks_lengths,
                            batch_inds,
                            object_labels=None):

        cfg = self.cfg
        # Batch weight at each point for loss (inverse of stacks_lengths for each point)
        min_len = tf.reduce_min(stacks_lengths, keepdims=True)
        batch_weights = tf.cast(min_len, tf.float32) / tf.cast(
            stacks_lengths, tf.float32)
        stacked_weights = tf.gather(batch_weights, batch_inds)

        # Starting radius of convolutions
        r_normal = cfg.first_subsampling_dl * cfg.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_batches_len = []

        # Loop over the blocks
        for block_i, block in enumerate(cfg.architecture):
            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or
                    'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * cfg.deform_radius / cfg.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = tf_batch_neighbors(stacked_points, stacked_points,
                                            stacks_lengths, stacks_lengths, r)
            else:
                # This layer only perform pooling, no neighbors required
                conv_i = tf.zeros((0, 1), dtype=tf.int32)

            # Pooling neighbors indices
            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / cfg.conv_radius

                # Subsampled points
                pool_p, pool_b = tf_batch_subsampling(stacked_points,
                                                      stacks_lengths, dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * cfg.deform_radius / cfg.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = tf_batch_neighbors(pool_p, stacked_points, pool_b,
                                            stacks_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = tf_batch_neighbors(stacked_points, pool_p,
                                          stacks_lengths, pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = tf.zeros((0, 1), dtype=tf.int32)
                pool_p = tf.zeros((0, 3), dtype=tf.float32)
                pool_b = tf.zeros((0,), dtype=tf.int32)
                up_i = tf.zeros((0, 1), dtype=tf.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            # TODO :
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            up_i = self.big_neighborhood_filter(up_i, len(input_points) + 1)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i]
            input_pools += [pool_i]
            input_upsamples += [up_i]
            input_batches_len += [stacks_lengths]

            # New points for next layer
            stacked_points = pool_p
            stacks_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        # Return inputs
        # Batch unstacking (with last layer indices for optional classif loss)
        stacked_batch_inds_0 = self.stack_batch_inds(input_batches_len[0])

        # Batch unstacking (with last layer indices for optional classif loss)
        stacked_batch_inds_1 = self.stack_batch_inds(input_batches_len[-1])

        if object_labels is None:

            # list of network inputs
            li = input_points + input_neighbors + input_pools + input_upsamples
            li += [
                stacked_features, stacked_weights, stacked_batch_inds_0,
                stacked_batch_inds_1
            ]
            li += [point_labels]

            return li

        else:

            # Object class ind for each point
            stacked_object_labels = tf.gather(object_labels, batch_inds)

            # list of network inputs
            li = input_points + input_neighbors + input_pools + input_upsamples
            li += [
                stacked_features, stacked_weights, stacked_batch_inds_0,
                stacked_batch_inds_1
            ]
            li += [point_labels, stacked_object_labels]

            return li

    def transform(self,
                  stacked_points,
                  stacked_colors,
                  point_labels,
                  stacks_lengths,
                  point_inds,
                  cloud_inds,
                  is_test=False):
        """[None, 3], [None, 3], [None], [None]"""
        cfg = self.cfg
        # Get batch indice for each point
        batch_inds = self.get_batch_inds(stacks_lengths)

        # Augment input points
        stacked_points, scales, rots = self.augment_input(
            stacked_points, batch_inds, is_test)

        # First add a column of 1 as feature for the network to be able to learn 3D shapes
        stacked_features = tf.ones((tf.shape(stacked_points)[0], 1),
                                   dtype=tf.float32)

        # Get coordinates and colors
        stacked_original_coordinates = stacked_colors[:, :3]
        stacked_colors = stacked_colors[:, 3:]

        # Augmentation : randomly drop colors
        if cfg.in_features_dim in [4, 5]:
            num_batches = batch_inds[-1] + 1
            s = tf.cast(
                tf.less(tf.random.uniform((num_batches,)), cfg.augment_color),
                tf.float32)
            stacked_s = tf.gather(s, batch_inds)
            stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)

        # Then use positions or not
        if cfg.in_features_dim == 1:
            pass
        elif cfg.in_features_dim == 2:
            stacked_features = tf.concat(
                (stacked_features, stacked_original_coordinates[:, 2:]), axis=1)
        elif cfg.in_features_dim == 3:
            stacked_features = stacked_colors
        elif cfg.in_features_dim == 4:
            stacked_features = tf.concat((stacked_features, stacked_colors),
                                         axis=1)
        elif cfg.in_features_dim == 5:
            stacked_features = tf.concat(
                (stacked_features, stacked_original_coordinates[:, 2:],
                 stacked_colors),
                axis=1)
        elif cfg.in_features_dim == 7:
            stacked_features = tf.concat(
                (stacked_features, stacked_original_coordinates,
                 stacked_colors),
                axis=1)
        else:
            raise ValueError(
                'Only accepted input dimensions are 1, 3, 4 and 7 (without and with rgb/xyz)'
            )

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points, stacked_features,
                                              point_labels, stacks_lengths,
                                              batch_inds)

        # Add scale and rotation for testing
        input_list += [scales, rots]
        input_list += [point_inds, cloud_inds]

        return input_list

    def inference_begin(self, data):
        attr = {'split': 'test'}
        self.inference_data = self.preprocess(data, attr)
        num_points = self.inference_data['search_tree'].data.shape[0]
        self.possibility = np.random.rand(num_points) * 1e-3
        self.test_probs = np.zeros(shape=[num_points, self.cfg.num_classes],
                                   dtype=np.float16)
        self.pbar = tqdm(total=self.possibility.shape[0])
        self.pbar_update = 0

    def inference_preprocess(self):
        flat_inputs, point_inds, stacks_lengths = self.transform_inference(
            self.inference_data)
        self.test_meta = {}
        self.test_meta['inds'] = point_inds
        self.test_meta['lens'] = stacks_lengths

        return flat_inputs

    def inference_end(self, results):
        results = tf.reshape(results, (-1, self.cfg.num_classes))
        results = tf.nn.softmax(results, axis=-1)
        results = results.cpu().numpy()
        test_smooth = 0.98

        inds = self.test_meta['inds']
        l = 0
        r = 0
        for len in self.test_meta['lens']:
            r += len
            self.test_probs[inds[l:r]] = self.test_probs[
                inds[l:r]] * test_smooth + (1 - test_smooth) * results[l:r]
            l += len

        self.pbar.update(self.possibility[self.possibility > 0.5].shape[0] -
                         self.pbar_update)
        self.pbar_update = self.possibility[self.possibility > 0.5].shape[0]

        if np.min(self.possibility) > 0.5:
            self.pbar.close()
            reproj_inds = self.inference_data['proj_inds']
            predict_scores = self.test_probs[reproj_inds]
            inference_result = {
                'predict_labels': np.argmax(predict_scores, 1),
                'predict_scores': predict_scores
            }

            self.inference_result = inference_result
            return True
        else:
            return False

    def transform_inference(self, data):
        cfg = self.cfg

        p_list = []
        c_list = []
        pl_list = []
        pi_list = []
        ci_list = []

        n_points = 0
        points = np.array(data['search_tree'].data)

        while (n_points < cfg.batch_limit):
            cloud_ind = 0
            point_ind = int(np.argmin(self.possibility))

            center_point = points[point_ind, :].reshape(1, -1)
            pick_point = center_point.copy()

            input_inds = data['search_tree'].query_radius(pick_point,
                                                          r=cfg.in_radius)[0]

            n = input_inds.shape[0]
            n_points += n

            dists = np.sum(np.square(
                (points[input_inds] - pick_point).astype(np.float32)),
                           axis=1)
            tuckeys = np.square(1 - dists / np.square(cfg.in_radius))
            tuckeys[dists > np.square(cfg.in_radius)] = 0
            self.possibility[input_inds] += tuckeys

            input_points = points[input_inds].copy() - pick_point
            feat = data['feat']

            t_normalize = self.cfg.get('t_normalize', {})
            input_points, feat = trans_normalize(input_points, feat,
                                                 t_normalize)

            if feat is None:
                coords = input_points
            else:
                coords = np.hstack((input_points, feat[input_inds]))

            coords[:, 2] += pick_point[:, 2]

            if len(data['label'][input_inds].shape) == 2:
                input_labels = data['label'][input_inds][:, 0]
            else:
                input_labels = data['label'][input_inds]

            if n > 0:
                p_list += [input_points]
                c_list += [coords]
                pl_list += [input_labels]
                pi_list += [input_inds]
                ci_list += [cloud_ind]

        stacked_points = np.concatenate(p_list, axis=0),
        stacked_colors = np.concatenate(c_list, axis=0),
        point_labels = np.concatenate(pl_list, axis=0),
        stacks_lengths = np.array([tp.shape[0] for tp in p_list],
                                  dtype=np.int32),
        point_inds = np.concatenate(pi_list, axis=0),
        cloud_inds = np.array(ci_list, dtype=np.int32)

        input_list = self.transform(
            tf.convert_to_tensor(np.array(stacked_points[0], dtype=np.float32)),
            tf.convert_to_tensor(np.array(stacked_colors[0], dtype=np.float32)),
            tf.convert_to_tensor(np.array(point_labels[0], dtype=np.int32)),
            tf.convert_to_tensor(np.array(stacks_lengths[0], dtype=np.int32)),
            tf.convert_to_tensor(np.array(point_inds[0], dtype=np.int32)),
            tf.convert_to_tensor(np.array(cloud_inds, dtype=np.int32)),
            is_test=True)
        return input_list, np.array(point_inds[0]), np.array(stacks_lengths[0])

    def preprocess(self, data, attr):
        cfg = self.cfg

        points = data['point'][:, 0:3]

        if 'label' not in data.keys() or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        split = attr['split']

        if 'feat' not in data.keys() or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        data = dict()

        if (feat is None):
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=cfg.first_subsampling_dl)
            sub_feat = None
        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                points,
                features=feat,
                labels=labels,
                grid_size=cfg.first_subsampling_dl)

        search_tree = KDTree(sub_points)

        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['search_tree'] = search_tree

        if split in ["test", "testing"]:
            proj_inds = np.squeeze(
                search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds

        return data

    def get_batch_gen(self, dataset, steps_per_epoch=None, batch_size=1):

        cfg = self.cfg
        if dataset.read_data(0)[0]['feat'] is None:
            dim_features = 3
        else:
            dim_features = 6

        def spatially_regular_gen():

            random_pick_n = None
            split = dataset.split

            epoch_n = dataset.num_pc if steps_per_epoch is None else steps_per_epoch

            batch_limit = cfg.batch_limit

            data_split = split

            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []

            batch_n = 0

            # Generator loop
            while (epoch_n):
                # Choose a random cloud
                cloud_ind = random.randint(0, dataset.num_pc - 1)

                data, attr = dataset.read_data(cloud_ind)

                point_ind = np.random.choice(len(data['point']), 1)

                points = data['point']

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.in_radius / 10,
                                         size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Indices of points in input region
                input_inds = data['search_tree'].query_radius(
                    pick_point, r=cfg.in_radius)[0]

                # Number collected
                n = input_inds.shape[0]

                # Safe check for very dense areas
                if n > batch_limit:
                    input_inds = np.random.choice(input_inds,
                                                  size=int(batch_limit) - 1,
                                                  replace=False)
                    n = input_inds.shape[0]

                # Collect points and colors
                input_points = points[input_inds].copy() - pick_point
                feat = data['feat']

                t_normalize = self.cfg.get('t_normalize', {})
                input_points, feat = trans_normalize(input_points, feat,
                                                     t_normalize)

                if feat is None:
                    coords = input_points.copy()
                else:
                    coords = np.hstack((input_points, feat[input_inds]))

                coords[:, 2] += pick_point[:, 2]

                if split in ['test', 'testing']:
                    input_labels = np.zeros(input_points.shape[0])
                else:
                    if len(data['label'][input_inds].shape) == 2:
                        input_labels = data['label'][input_inds][:, 0]
                    else:
                        input_labels = data['label'][input_inds]

                # In case batch is full, yield it and reset it
                if batch_n + n > batch_limit and batch_n > 0:
                    epoch_n -= 1

                    yield (np.concatenate(p_list,
                                          axis=0), np.concatenate(c_list,
                                                                  axis=0),
                           np.concatenate(pl_list, axis=0),
                           np.array([tp.shape[0] for tp in p_list]),
                           np.concatenate(pi_list, axis=0),
                           np.array(ci_list, dtype=np.int32))

                    p_list = []
                    c_list = []
                    pl_list = []
                    pi_list = []
                    ci_list = []
                    batch_n = 0

                # Add data to current batch
                if n > 0:
                    p_list += [input_points]
                    c_list += [coords]
                    pl_list += [input_labels]
                    pi_list += [input_inds]
                    ci_list += [cloud_ind]

                # Update batch size
                batch_n += n

            if batch_n > 0:
                yield (np.concatenate(p_list,
                                      axis=0), np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list,
                                      axis=0), np.array(ci_list,
                                                        dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32,
                     tf.int32)
        gen_shapes = ([None, 3], [None,
                                  dim_features], [None], [None], [None], [None])

        return gen_func, gen_types, gen_shapes


MODEL._register_module(KPFCNN, 'tf')


def tf_batch_neighbors(queries, points, q_batches, p_batches, r):
    nns = FixedRadiusSearch()
    q_splits = tf.cast(
        tf.concat([tf.constant([0]), tf.cumsum(q_batches)], axis=0), tf.int64)
    p_splits = tf.cast(
        tf.concat([tf.constant([0]), tf.cumsum(p_batches)], axis=0), tf.int64)
    result = nns(points, queries, r, p_splits, q_splits)

    idx = result.neighbors_index
    splits = result.neighbors_row_splits

    ragged_idx = tf.RaggedTensor.from_row_splits(idx, splits)
    dense_idx = ragged_idx.to_tensor(default_value=tf.shape(points)[0])

    return dense_idx
