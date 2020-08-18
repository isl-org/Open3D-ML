from os import makedirs
from os.path import exists
import time
import tensorflow as tf
import numpy as np
import random
from os.path import exists, join, isfile, dirname, abspath, split
import sys
from pathlib import Path
from sklearn.neighbors import KDTree

from ...datasets.utils.dataprocessing import DataProcessing

# Convolution functions
# import network_blocks
from .network_blocks import assemble_FCNN_blocks, segmentation_head, multi_segmentation_head
from .network_blocks import segmentation_loss, multi_segmentation_loss

# Load custom operation
BASE_DIR = Path(abspath(__file__))

tf_neighbors_module = tf.load_op_library(
    str(BASE_DIR.parent.parent / 'utils' / 'tf_custom_ops' /
        'tf_neighbors.so'))
tf_batch_neighbors_module = tf.load_op_library(
    str(BASE_DIR.parent.parent / 'utils' / 'tf_custom_ops' /
        'tf_batch_neighbors.so'))
tf_subsampling_module = tf.load_op_library(
    str(BASE_DIR.parent.parent / 'utils' / 'tf_custom_ops' /
        'tf_subsampling.so'))
tf_batch_subsampling_module = tf.load_op_library(
    str(BASE_DIR.parent.parent / 'utils' / 'tf_custom_ops' /
        'tf_batch_subsampling.so'))


def tf_batch_subsampling(points, batches_len, sampleDl):
    return tf_batch_subsampling_module.batch_grid_subsampling(
        points, batches_len, sampleDl)


def tf_batch_neighbors(queries, supports, q_batches, s_batches, radius):
    return tf_batch_neighbors_module.batch_ordered_neighbors(
        queries, supports, q_batches, s_batches, radius)


class KPFCNN:
    def __init__(self, cfg):
        # Model parameters
        self.cfg = cfg

        # From config parameter, compute higher bound of neighbors number in a neighborhood
        hist_n = int(np.ceil(4 / 3 * np.pi * (cfg.density_parameter + 1) ** 3))

        # Initiate neighbors limit with higher bound
        self.neighborhood_limits = np.full(cfg.num_layers, hist_n, dtype=np.int32)

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """
        # crop neighbors matrix
        return neighbors[:, :self.neighborhood_limits[layer]]

    def init_input(flat_inputs):
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path == None:
                self.saving_path = time.strftime(
                    'results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            if not exists(self.saving_path):
                makedirs(self.saving_path)

        ########
        # Inputs
        ########

        # Sort flatten inputs in a dictionary
        with tf.variable_scope('inputs'):
            self.inputs = dict()
            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 *
                                                   config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 *
                                               config.num_layers]
            self.inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 *
                                                   config.num_layers]
            ind = 4 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_labels'] = flat_inputs[ind]
            ind += 1
            self.labels = self.inputs['point_labels']

            if config.network_model in [
                    'multi_segmentation', 'multi_cloud_segmentation'
            ]:
                self.inputs['super_labels'] = flat_inputs[ind]
                ind += 1

            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]

            if config.network_model in [
                    "cloud_segmentation", 'multi_cloud_segmentation'
            ]:
                ind += 1
                self.inputs['point_inds'] = flat_inputs[ind]
                ind += 1
                self.inputs['cloud_inds'] = flat_inputs[ind]

            elif config.network_model in [
                    'multi_segmentation', 'segmentation'
            ]:
                ind += 1
                self.inputs['object_inds'] = flat_inputs[ind]

            # Dropout placeholder
            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        ########
        # Layers
        ########

        # Create layers
        with tf.variable_scope('KernelPointNetwork'):
            output_features = assemble_FCNN_blocks(self.inputs, self.config,
                                                   self.dropout_prob)

            if config.network_model in [
                    "multi_segmentation", 'multi_cloud_segmentation'
            ]:
                self.logits = multi_segmentation_head(
                    output_features, self.inputs['super_labels'], self.config,
                    self.dropout_prob)
            else:
                self.logits = segmentation_head(output_features, self.config,
                                                self.dropout_prob)

        ########
        # Losses
        ########

        with tf.variable_scope('loss'):

            if config.network_model in [
                    "multi_segmentation", 'multi_cloud_segmentation'
            ]:
                self.output_loss = multi_segmentation_loss(
                    self.logits,
                    self.inputs,
                    batch_average=self.config.batch_averaged_loss)

            elif len(self.config.ignored_label_inds) > 0:

                # Boolean mask of points that should be ignored
                ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
                for ign_label in self.config.ignored_label_inds:
                    ignored_bool = tf.logical_or(
                        ignored_bool, tf.equal(self.labels, ign_label))

                # Collect logits and labels that are not ignored
                inds = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
                new_logits = tf.gather(self.logits, inds, axis=0)
                new_dict = {
                    'point_labels': tf.gather(self.labels, inds, axis=0)
                }

                # Reduce label values in the range of logit shape
                reducing_list = tf.range(self.config.num_classes,
                                         dtype=tf.int32)
                inserted_value = tf.zeros((1, ), dtype=tf.int32)
                for ign_label in self.config.ignored_label_inds:
                    reducing_list = tf.concat([
                        reducing_list[:ign_label], inserted_value,
                        reducing_list[ign_label:]
                    ], 0)
                new_dict['point_labels'] = tf.gather(reducing_list,
                                                     new_dict['point_labels'])

                # Add batch weigths to dict if needed
                if self.config.batch_averaged_loss:
                    new_dict['batch_weights'] = self.inputs['batch_weights']

                # Output loss
                self.output_loss = segmentation_loss(
                    new_logits,
                    new_dict,
                    batch_average=self.config.batch_averaged_loss)

            else:
                self.output_loss = segmentation_loss(
                    self.logits,
                    self.inputs,
                    batch_average=self.config.batch_averaged_loss)

            # Add regularization
            self.loss = self.regularization_losses() + self.output_loss

        return

    def regularization_losses(self):

        #####################
        # Regularization loss
        #####################

        # Get L2 norm of all weights
        regularization_losses = [
            tf.nn.l2_loss(v) for v in tf.global_variables()
            if 'weights' in v.name
        ]
        self.regularization_loss = self.config.weights_decay * tf.add_n(
            regularization_losses)

        ##############################
        # Gaussian regularization loss
        ##############################

        gaussian_losses = []
        for v in tf.global_variables():
            if 'kernel_extents' in v.name:

                # Layer index
                layer = int(v.name.split('/')[1].split('_')[-1])

                # Radius of convolution for this layer
                conv_radius = cfg.first_subsampling_dl * self.config.density_parameter * (
                    2**(layer - 1))

                # Target extent
                target_extent = conv_radius / 1.5
                gaussian_losses += [tf.nn.l2_loss(v - target_extent)]

        if len(gaussian_losses) > 0:
            self.gaussian_loss = self.config.gaussian_decay * tf.add_n(
                gaussian_losses)
        else:
            self.gaussian_loss = tf.constant(0, dtype=tf.float32)

        #############################
        # Offsets regularization loss
        #############################

        offset_losses = []

        if self.config.offsets_loss == 'permissive':

            for op in tf.get_default_graph().get_operations():
                if op.name.endswith('deformed_KP'):

                    # Get deformed positions
                    deformed_positions = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    conv_radius = cfg.first_subsampling_dl * self.config.density_parameter * (
                        2**layer)

                    # Normalized KP locations
                    KP_locs = deformed_positions / conv_radius

                    # Loss will be zeros inside radius and linear outside radius
                    # Mean => loss independent from the number of input points
                    radius_outside = tf.maximum(0.0,
                                                tf.norm(KP_locs, axis=2) - 1.0)
                    offset_losses += [tf.reduce_mean(radius_outside)]

        elif self.config.offsets_loss == 'fitting':

            for op in tf.get_default_graph().get_operations():

                if op.name.endswith('deformed_d2'):

                    # Get deformed distances
                    deformed_d2 = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    KP_extent = cfg.first_subsampling_dl * cfg.KP_extent * (
                        2**layer)

                    # Get the distance to closest input point
                    KP_min_d2 = tf.reduce_min(deformed_d2, axis=1)

                    # Normalize KP locations to be independant from layers
                    KP_min_d2 = KP_min_d2 / (KP_extent**2)

                    # Loss will be the square distance to closest input point.
                    # Mean => loss independent from the number of input points
                    offset_losses += [tf.reduce_mean(KP_min_d2)]

                if op.name.endswith('deformed_KP'):

                    # Get deformed positions
                    deformed_KP = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    KP_extent = cfg.first_subsampling_dl * cfg.KP_extent * (
                        2**layer)

                    # Normalized KP locations
                    KP_locs = deformed_KP / KP_extent

                    # Point should not be close to each other
                    for i in range(cfg.num_kernel_points):
                        other_KP = tf.stop_gradient(
                            tf.concat(
                                [KP_locs[:, :i, :], KP_locs[:, i + 1:, :]],
                                axis=1))
                        distances = tf.sqrt(
                            tf.reduce_sum(tf.square(other_KP -
                                                    KP_locs[:, i:i + 1, :]),
                                          axis=2))
                        repulsive_losses = tf.reduce_sum(tf.square(
                            tf.maximum(0.0, 1.5 - distances)),
                                                         axis=1)
                        offset_losses += [tf.reduce_mean(repulsive_losses)]

        elif self.config.offsets_loss != 'none':
            raise ValueError('Unknown offset loss')

        if len(offset_losses) > 0:
            self.offsets_loss = self.config.offsets_decay * tf.add_n(
                offset_losses)
        else:
            self.offsets_loss = tf.constant(0, dtype=tf.float32)

        return self.offsets_loss + self.gaussian_loss + self.regularization_loss

    def parameters_log(self):

        self.config.save(self.saving_path)

    def get_batch_inds(self, stacks_len):
        """
        Method computing the batch indices of all points, given the batch element sizes (stack lengths). Example:
        From [3, 2, 5], it would return [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
        """

        # Initiate batch inds tensor
        num_batches = tf.shape(stacks_len)[0]
        num_points = tf.reduce_sum(stacks_len)
        batch_inds_0 = tf.zeros((num_points, ), dtype=tf.int32)

        # Define body of the while loop
        def body(batch_i, point_i, b_inds):

            num_in = stacks_len[batch_i]
            num_before = tf.cond(tf.less(batch_i, 1), lambda: tf.zeros(
                (), dtype=tf.int32),
                                 lambda: tf.reduce_sum(stacks_len[:batch_i]))
            num_after = tf.cond(
                tf.less(batch_i, num_batches - 1),
                lambda: tf.reduce_sum(stacks_len[batch_i + 1:]),
                lambda: tf.zeros((), dtype=tf.int32))

            # Update current element indices
            inds_before = tf.zeros((num_before, ), dtype=tf.int32)
            inds_in = tf.fill((num_in, ), batch_i)
            inds_after = tf.zeros((num_after, ), dtype=tf.int32)
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

    def augment_input(self, stacked_points, batch_inds):

        cfg = self.cfg
        # Parameter
        num_batches = batch_inds[-1] + 1

        ##########
        # Rotation
        ##########

        if cfg.augment_rotation == 'vertical':

            # Choose a random angle for each element
            theta = tf.random.uniform((num_batches, ),
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
            stacked_points = tf.reshape(
                tf.matmul(tf.expand_dims(stacked_points, axis=1),
                          stacked_rots), [-1, 3])

        elif cfg.augment_rotation == 'none':
            R = tf.eye(3, batch_shape=(num_batches, ))

        else:
            raise ValueError('Unknown rotation augmentation : ' +
                             cfg.augment_rotation)

        #######
        # Scale
        #######

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

        # Apply scales
        stacked_points = stacked_points * stacked_scales

        #######
        # Noise
        #######

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

        # KPConv specific parameters
        density_parameter = 5.0

        # Starting radius of convolutions
        r_normal = cfg.first_subsampling_dl * cfg.KP_extent * 2.5

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_batches_len = []

        ######################
        # Loop over the blocks
        ######################

        for block_i, block in enumerate(cfg.architecture):

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block):
                layer_blocks += [block]
                if block_i < len(cfg.architecture) - 1 and not (
                        'upsample' in cfg.architecture[block_i + 1]):
                    continue

            # Convolution neighbors indices
            # *****************************

            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck
                           for blck in layer_blocks[:-1]]):
                    r = r_normal * density_parameter / (cfg.KP_extent * 2.5)
                else:
                    r = r_normal
                conv_i = tf_batch_neighbors(stacked_points, stacked_points,
                                            stacks_lengths, stacks_lengths, r)
            else:
                # This layer only perform pooling, no neighbors required
                conv_i = tf.zeros((0, 1), dtype=tf.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / (cfg.KP_extent * 2.5)

                # Subsampled points
                pool_p, pool_b = tf_batch_subsampling(stacked_points,
                                                      stacks_lengths,
                                                      sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * density_parameter / (cfg.KP_extent * 2.5)
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
                pool_b = tf.zeros((0, ), dtype=tf.int32)
                up_i = tf.zeros((0, 1), dtype=tf.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            # TODO :
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            up_i = self.big_neighborhood_filter(up_i, len(input_points))

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

        ###############
        # Return inputs
        ###############

        # Batch unstacking (with last layer indices for optionnal classif loss)
        stacked_batch_inds_0 = self.stack_batch_inds(input_batches_len[0])

        # Batch unstacking (with last layer indices for optionnal classif loss)
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

    def transform(self, stacked_points, stacked_colors, point_labels,
                  stacks_lengths, point_inds, cloud_inds):
        """
        [None, 3], [None, 3], [None], [None]
        """
        cfg = self.cfg
        # Get batch indice for each point
        batch_inds = self.get_batch_inds(stacks_lengths)

        # Augment input points
        stacked_points, scales, rots = self.augment_input(
            stacked_points, batch_inds)

        # First add a column of 1 as feature for the network to be able to learn 3D shapes
        stacked_features = tf.ones((tf.shape(stacked_points)[0], 1),
                                   dtype=tf.float32)

        # Get coordinates and colors
        stacked_original_coordinates = stacked_colors[:, 3:]
        stacked_colors = stacked_colors[:, :3]

        # Augmentation : randomly drop colors
        if cfg.in_features_dim in [4, 5]:
            num_batches = batch_inds[-1] + 1
            s = tf.cast(
                tf.less(tf.random.uniform((num_batches, )), cfg.augment_color),
                tf.float32)
            stacked_s = tf.gather(s, batch_inds)
            stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)

        # Then use positions or not
        if cfg.in_features_dim == 1:
            pass
        elif cfg.in_features_dim == 2:
            stacked_features = tf.concat(
                (stacked_features, stacked_original_coordinates[:, 2:]),
                axis=1)
        elif cfg.in_features_dim == 3:
            stacked_features = stacked_colors
        elif cfg.in_features_dim == 4:
            stacked_features = tf.concat((stacked_features, stacked_colors),
                                         axis=1)
        elif cfg.in_features_dim == 5:
            stacked_features = tf.concat((stacked_features, stacked_colors,
                                          stacked_original_coordinates[:, 2:]),
                                         axis=1)
        elif cfg.in_features_dim == 7:
            stacked_features = tf.concat(
                (stacked_features, stacked_colors, stacked_points), axis=1)
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

    def preprocess(self, data, attr):
        cfg = self.cfg
        if 'feat' not in data.keys():
            data['feat'] = None

        points = data['point'][:, 0:3]
        feat = data['feat'][:, 0:3]
        labels = data['label']
        split = attr['split']

        if (feat is None):
            sub_feat = None

        data = dict()

        if (feat is None):
            sub_points, sub_labels = DataProcessing.grid_sub_sampling(
                points, labels=labels, grid_size=cfg.first_subsampling_dl)

        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_sub_sampling(
                points,
                features=feat,
                labels=labels,
                grid_size=cfg.first_subsampling_dl)

        search_tree = KDTree(sub_points)

        data['point'] = np.array(sub_points)
        data['feat'] = np.array(sub_feat)
        data['label'] = np.array(sub_labels)
        data['search_tree'] = search_tree

        if split != "training":
            proj_inds = np.squeeze(
                search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds

        return data

    def crop_pc(self, points, feat, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        num_points = 65536
        if (points.shape[0] < num_points):
            select_idx = np.array(range(points.shape[0]))
            diff = num_points - points.shape[0]
            select_idx = list(select_idx) + list(
                random.choices(select_idx, k=diff))
            random.shuffle(select_idx)
        else:
            center_point = points[pick_idx, :].reshape(1, -1)
            select_idx = search_tree.query(center_point, k=num_points)[1][0]

        # select_idx = DataProcessing.shuffle_idx(select_idx)
        random.shuffle(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        if (feat is None):
            select_feat = None
        else:
            select_feat = feat[select_idx]
        return select_points, select_feat, select_labels, select_idx

    def get_batch_gen(self, dataset):

        cfg = self.cfg

        def spatially_regular_gen():

            random_pick_n = None
            epoch_n = cfg.epoch_steps * cfg.batch_num
            split = dataset.split

            batch_limit = 5000  # TODO : read from calibrate_batch, typically 100 * batch_size required

            # Initiate potentials for regular generation
            if not hasattr(self, 'potentials'):
                self.potentials = {}
                self.min_potentials = {}

            # Reset potentials
            self.potentials[split] = []
            self.min_potentials[split] = []
            data_split = split

            #TODO :
            # for i, tree in enumerate(self.input_trees[data_split]):
            #     self.potentials[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            #     self.min_potentials[split] += [float(np.min(self.potentials[split][-1]))]

            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []

            batch_n = 0

            # Generator loop
            for i in range(epoch_n):
                # Choose a random cloud
                # cloud_ind = int(np.argmin(self.min_potentials[split]))
                cloud_ind = random.randint(0, dataset.num_pc - 1)

                data, attr = dataset.read_data(cloud_ind)

                # Choose point ind as minimum of potentials
                # point_ind = np.argmin(self.potentials[split][cloud_ind])
                point_ind = np.random.choice(len(data['point']), 1)

                # Get points from tree structure
                # points = np.array(self.input_trees[data_split][cloud_ind].data, copy=False)
                points = np.array(data['search_tree'].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)
                # Add noise to the center point
                # if split != 'ERF':
                #     noise = np.random.normal(scale=config.in_radius/10, size=center_point.shape)
                #     pick_point = center_point + noise.astype(center_point.dtype)
                # else:
                #     pick_point = center_point
                pick_point = center_point

                # Indices of points in input region
                # input_inds = self.input_trees[data_split][cloud_ind].query_radius(pick_point,
                #                                                                 r=config.in_radius)[0]
                input_inds = data['search_tree'].query_radius(
                    pick_point, r=cfg.in_radius)[0]

                # Number collected
                n = input_inds.shape[0]

                # Update potentials (Tuckey weights)
                # if split != 'ERF':
                #     dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                #     tukeys = np.square(1 - dists / np.square(in_radius))
                #     tukeys[dists > np.square(in_radius)] = 0
                #     self.potentials[split][cloud_ind][input_inds] += tukeys
                #     self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

                # Safe check for very dense areas
                if n > batch_limit:
                    input_inds = np.random.choice(input_inds,
                                                  size=int(batch_limit) - 1,
                                                  replace=False)
                    n = input_inds.shape[0]

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(
                    np.float32)
                # input_colors = self.input_colors[data_split][cloud_ind][input_inds]
                input_colors = data['feat'][input_inds]

                if split in ['test']:
                    input_labels = np.zeros(input_points.shape[0])
                else:
                    # input_labels = self.input_labels[data_split][cloud_ind][input_inds]
                    input_labels = data['label'][input_inds][:, 0]
                    # input_labels = np.array([self.label_to_idx[l] for l in input_labels])

                # In case batch is full, yield it and reset it
                if batch_n + n > batch_limit and batch_n > 0:

                    yield (np.concatenate(p_list, axis=0),
                           np.concatenate(c_list, axis=0),
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
                    c_list += [
                        np.hstack((input_colors, input_points + pick_point))
                    ]
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
        gen_shapes = ([None, 3], [None, 6], [None], [None], [None], [None])

        return gen_func, gen_types, gen_shapes

