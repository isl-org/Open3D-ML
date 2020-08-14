from abc import abstractmethod
from tqdm import tqdm
import random

import tensorflow as tf
import numpy as np
from ml3d.torch.utils import dataset_helper

from ml3d.datasets.utils import DataProcessing
from sklearn.neighbors import KDTree


# def randlanet_preprocess():

class TF_Dataset():
    def __init__(self,
                 *args,
                 dataset=None,
                 preprocess=None,
                 transform=None,
                 no_progress: bool = False,
                 **kwargs):
        self.dataset = dataset
        self.preprocess = preprocess
        self.transform = transform
        if preprocess is not None:
            cache_dir = getattr(dataset.cfg, 'cache_dir')
            assert cache_dir is not None, 'cache directory is not given'

            self.cache_convert = dataset_helper.Cache(
                preprocess,
                cache_dir=cache_dir,
                cache_key=dataset_helper._get_hash(repr(preprocess)[:-15]))

            uncached = [
                idx for idx in range(len(dataset))
                if dataset.get_attr(idx)['name'] not in
                self.cache_convert.cached_ids
            ]
            if len(uncached) > 0:
                for idx in tqdm(
                        range(len(dataset)), desc='preprocess', disable=no_progress):
                    attr = dataset.get_attr(idx)
                    data = dataset.get_data(idx)
                    name = attr['name']

                    self.cache_convert(name, data, attr)

        else:
            self.cache_convert = None

        self.epoch_n = 10 * 500 # TODO : number of batches * steps per epoch
        self.num_threads = 3 # read from config
        self.split = dataset.split
        self.pc_list = dataset.path_list
        self.num_pc = len(self.pc_list)

    # def generator()
    def read_data(self, key):
        attr = self.dataset.get_attr(key)
        # print(attr)
        if self.cache_convert is None:
            data = self.dataset.get_data(key)
        else:
            data = self.cache_convert(attr['name'])

        pick_idx = np.random.choice(len(data['point']), 1)
        pc, feat, label, _ = crop_pc(data['point'], data['feat'], data['label'], data['search_tree'], pick_idx)

        return pc, feat, label

    def tf_get_batch_inds(self, stacks_len):
        """
        Method computing the batch indices of all points, given the batch element sizes (stack lengths). Example:
        From [3, 2, 5], it would return [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
        """

        # Initiate batch inds tensor
        num_batches = tf.shape(stacks_len)[0]
        num_points = tf.reduce_sum(stacks_len)
        batch_inds_0 = tf.zeros((num_points,), dtype=tf.int32)

        # Define body of the while loop
        def body(batch_i, point_i, b_inds):

            num_in = stacks_len[batch_i]
            num_before = tf.cond(tf.less(batch_i, 1),
                                    lambda: tf.zeros((), dtype=tf.int32),
                                    lambda: tf.reduce_sum(stacks_len[:batch_i]))
            num_after = tf.cond(tf.less(batch_i, num_batches - 1),
                                lambda: tf.reduce_sum(stacks_len[batch_i+1:]),
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
                                            shape_invariants=[tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None])])

        return batch_inds


    def tf_augment_input(self, stacked_points, batch_inds, config):

        augment_scale_anisotropic = True
        augment_symmetries = [True, False, False]
        augment_rotation = 'vertical'
        augment_scale_min = 0.8
        augment_scale_max = 1.2
        augment_noise = 0.001
        augment_occlusion = 'none'
        augment_color = 0.8
        # Parameter
        num_batches = batch_inds[-1] + 1

        ##########
        # Rotation
        ##########

        if augment_rotation == 'vertical':

            # Choose a random angle for each element
            theta = tf.random_uniform((num_batches,), minval=0, maxval=2*np.pi)

            # Rotation matrices
            c, s = tf.cos(theta), tf.sin(theta)
            cs0 = tf.zeros_like(c)
            cs1 = tf.ones_like(c)
            R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
            R = tf.reshape(R, (-1, 3, 3))

            # Create N x 3 x 3 rotation matrices to multiply with stacked_points
            stacked_rots = tf.gather(R, batch_inds)

            # Apply rotations
            stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=1), stacked_rots), [-1, 3])

        elif augment_rotation == 'none':
            R = tf.eye(3, batch_shape=(num_batches,))

        else:
            raise ValueError('Unknown rotation augmentation : ' + augment_rotation)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = augment_scale_min
        max_s = augment_scale_max

        if augment_scale_anisotropic:
            s = tf.random_uniform((num_batches, 3), minval=min_s, maxval=max_s)
        else:
            s = tf.random_uniform((num_batches, 1), minval=min_s, maxval=max_s)

        symmetries = []
        for i in range(3):
            if augment_symmetries[i]:
                symmetries.append(tf.round(tf.random_uniform((num_batches, 1))) * 2 - 1)
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

        noise = tf.random_normal(tf.shape(stacked_points), stddev=augment_noise)
        stacked_points = stacked_points + noise

        return stacked_points, s, R

    def tf_segmentation_inputs(self,
                               config,
                               stacked_points,
                               stacked_features,
                               point_labels,
                               stacks_lengths,
                               batch_inds,
                               object_labels=None):

        # Batch weight at each point for loss (inverse of stacks_lengths for each point)
        min_len = tf.reduce_min(stacks_lengths, keep_dims=True)
        batch_weights = tf.cast(min_len, tf.float32) / tf.cast(stacks_lengths, tf.float32)
        stacked_weights = tf.gather(batch_weights, batch_inds)

        architecture = ['simple',
                        'resnetb',
                        'resnetb_strided',
                        'resnetb',
                        'resnetb_strided',
                        'resnetb',
                        'resnetb_strided',
                        'resnetb',
                        'resnetb_strided',
                        'resnetb',
                        'nearest_upsample',
                        'unary',
                        'nearest_upsample',
                        'unary',
                        'nearest_upsample',
                        'unary',
                        'nearest_upsample',
                        'unary']

        # KPConv specific parameters
        num_kernel_points = 15
        first_subsampling_dl = 0.04
        KP_influence = 'linear'
        KP_extent = 1.0
        density_parameter = 5.0

        # Starting radius of convolutions
        r_normal = first_subsampling_dl * KP_extent * 2.5

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

        for block_i, block in enumerate(architecture):

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block):
                layer_blocks += [block]
                if block_i < len(architecture) - 1 and not ('upsample' in architecture[block_i + 1]):
                    continue

            # Convolution neighbors indices
            # *****************************

            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                    r = r_normal * density_parameter / (KP_extent * 2.5)
                else:
                    r = r_normal
                conv_i = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
            else:
                # This layer only perform pooling, no neighbors required
                conv_i = tf.zeros((0, 1), dtype=tf.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / (KP_extent * 2.5)

                # Subsampled points
                pool_p, pool_b = tf_batch_subsampling(stacked_points, stacks_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * density_parameter / (KP_extent * 2.5)
                else:
                    r = r_normal

                # Subsample indices
                pool_i = tf_batch_neighbors(pool_p, stacked_points, pool_b, stacks_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = tf_batch_neighbors(stacked_points, pool_p, stacks_lengths, pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = tf.zeros((0, 1), dtype=tf.int32)
                pool_p = tf.zeros((0, 3), dtype=tf.float32)
                pool_b = tf.zeros((0,), dtype=tf.int32)
                up_i = tf.zeros((0, 1), dtype=tf.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
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
        stacked_batch_inds_0 = self.tf_stack_batch_inds(input_batches_len[0])

        # Batch unstacking (with last layer indices for optionnal classif loss)
        stacked_batch_inds_1 = self.tf_stack_batch_inds(input_batches_len[-1])

        if object_labels is None:

            # list of network inputs
            li = input_points + input_neighbors + input_pools + input_upsamples
            li += [stacked_features, stacked_weights, stacked_batch_inds_0, stacked_batch_inds_1]
            li += [point_labels]

            return li

        else:

            # Object class ind for each point
            stacked_object_labels = tf.gather(object_labels, batch_inds)

            # list of network inputs
            li = input_points + input_neighbors + input_pools + input_upsamples
            li += [stacked_features, stacked_weights, stacked_batch_inds_0, stacked_batch_inds_1]
            li += [point_labels, stacked_object_labels]

            return li

    def tf_map(self, stacked_points, stacked_colors, point_labels, stacks_lengths, point_inds, cloud_inds):
        """
        [None, 3], [None, 3], [None], [None]
        """
        in_features_dim = 5 # TODO : read from config
        augment_color = 0.8

        # Get batch indice for each point
        batch_inds = self.tf_get_batch_inds(stacks_lengths)

        # Augment input points
        stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                batch_inds,
                                                                config)

        # First add a column of 1 as feature for the network to be able to learn 3D shapes
        stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

        # Get coordinates and colors
        stacked_original_coordinates = stacked_colors[:, 3:]
        stacked_colors = stacked_colors[:, :3]

        # Augmentation : randomly drop colors
        if in_features_dim in [4, 5]:
            num_batches = batch_inds[-1] + 1
            s = tf.cast(tf.less(tf.random_uniform((num_batches,)), config.augment_color), tf.float32)
            stacked_s = tf.gather(s, batch_inds)
            stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)

        # Then use positions or not
        if in_features_dim == 1:
            pass
        elif in_features_dim == 2:
            stacked_features = tf.concat((stacked_features, stacked_original_coordinates[:, 2:]), axis=1)
        elif in_features_dim == 3:
            stacked_features = stacked_colors
        elif in_features_dim == 4:
            stacked_features = tf.concat((stacked_features, stacked_colors), axis=1)
        elif in_features_dim == 5:
            stacked_features = tf.concat((stacked_features, stacked_colors, stacked_original_coordinates[:, 2:]), axis=1)
        elif in_features_dim == 7:
            stacked_features = tf.concat((stacked_features, stacked_colors, stacked_points), axis=1)
        else:
            raise ValueError('Only accepted input dimensions are 1, 3, 4 and 7 (without and with rgb/xyz)')

        # Get the whole input list
        input_list = self.tf_segmentation_inputs(config,
                                                    stacked_points,
                                                    stacked_features,
                                                    point_labels,
                                                    stacks_lengths,
                                                    batch_inds)

        # Add scale and rotation for testing
        input_list += [scales, rots]
        input_list += [point_inds, cloud_inds]

        return input_list


    def spatially_regular_gen(self):
        random_pick_n = None
        epoch_n = self.epoch_n
        split = self.split

        # TODO : read from config
        in_radius = 2.0
        batch_limit = 500 # read from calibrate_batch

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
            cloud_ind = random.randint(0, self.num_pc - 1)
            
            attr = self.dataset.get_attr(cloud_ind)
            if self.cache_convert is None:
                data = self.dataset.get_data(cloud_ind)
            else:
                data = self.cache_convert(attr['name'])


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
            input_inds = data['search_tree'].query_radius(pick_point, r = in_radius)[0]

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
                input_inds = np.random.choice(input_inds, size=int(batch_limit)-1, replace=False)
                n = input_inds.shape[0]

            # Collect points and colors
            input_points = (points[input_inds] - pick_point).astype(np.float32)
            # input_colors = self.input_colors[data_split][cloud_ind][input_inds]
            input_colors = data['feat'][input_inds]

            if split in ['test']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                # input_labels = self.input_labels[data_split][cloud_ind][input_inds]
                input_labels = data['label'][input_inds][:, 0]
                # input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            # In case batch is full, yield it and reset it
            print("iter\n\n\n")
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
                c_list += [np.hstack((input_colors, input_points + pick_point))]
                pl_list += [input_labels]
                pi_list += [input_inds]
                ci_list += [cloud_ind]

            # Update batch size
            batch_n += n

        if batch_n > 0:
            yield (np.concatenate(p_list, axis=0),
                    np.concatenate(c_list, axis=0),
                    np.concatenate(pl_list, axis=0),
                    np.array([tp.shape[0] for tp in p_list]),
                    np.concatenate(pi_list, axis=0),
                    np.array(ci_list, dtype=np.int32))


    def get_loader(self):

        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 6], [None], [None], [None], [None])

        tf_dataset = tf.data.Dataset.from_generator(self.spatially_regular_gen, gen_types, gen_shapes)
        # tf_dataset = tf.data.Dataset.range(len(self.dataset))
        # tf_dataset = tf_dataset.map(lambda x : tf.numpy_function(func = self.read_data, inp = [x], Tout = [tf.float32, tf.float32,
        #                             tf.int32]))

        # tf_dataset = tf_dataset.map(map_func = self.transform)
        tf_dataset = tf_dataset.map(map_func=self.tf_map, num_parallel_calls=self.num_threads)


        return tf_dataset



def kpconv_preprocess(data, attr):
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
            points, labels=labels, grid_size=0.06)

    else:
        sub_points, sub_feat, sub_labels = DataProcessing.grid_sub_sampling(
            points, features=feat, labels=labels, grid_size=0.06)

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



def crop_pc(points, feat, labels, search_tree, pick_idx):
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
        select_idx = search_tree.query(center_point,
                                        k=num_points)[1][0]

    # select_idx = DataProcessing.shuffle_idx(select_idx)
    random.shuffle(select_idx)
    select_points = points[select_idx]
    select_labels = labels[select_idx]
    if (feat is None):
        select_feat = None
    else:
        select_feat = feat[select_idx]
    return select_points, select_feat, select_labels, select_idx


def randlanet_transform(pc, feat, label):
    num_layers = 5
    k_n = 16
    sub_sampling_ratio = [4, 4, 4, 4, 2]

    if (feat is not None):
        features = tf.concat([pc, feat], axis=1)
    else:
        features = pc

    input_points = []
    input_neighbors = []
    input_pools = []
    input_up_samples = []

    for i in range(num_layers):
        neighbour_idx = tf.py_function(DataProcessing.knn_search, [pc, pc, k_n], tf.int32)

        sub_points = pc[:tf.shape(pc)[0] // sub_sampling_ratio[i], :]
        pool_i = neighbour_idx[:tf.shape(pc)[0] //
                                sub_sampling_ratio[i], :]
        up_i = tf.py_function(DataProcessing.knn_search, [sub_points, pc, 1], tf.int32)
        input_points.append(pc)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)
        pc = sub_points

    input_list = input_points + input_neighbors + input_pools + input_up_samples
    input_list += [features, label]

    return input_list


def randlanet_preprocess(data, attr):
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
            points, labels=labels, grid_size=0.06)

    else:
        sub_points, sub_feat, sub_labels = DataProcessing.grid_sub_sampling(
            points, features=feat, labels=labels, grid_size=0.06)

    search_tree = KDTree(sub_points)

    data['point'] = sub_points
    data['feat'] = sub_feat
    data['label'] = sub_labels
    data['search_tree'] = search_tree

    if split != "training":
        proj_inds = np.squeeze(
            search_tree.query(points, return_distance=False))
        proj_inds = proj_inds.astype(np.int32)
        data['proj_inds'] = proj_inds

    return data



from ml3d.torch.utils import Config
from ml3d.datasets import Toronto3D

if __name__ == '__main__':
    config = '../../torch/configs/randlanet_toronto3d.py'
    cfg = Config.load_from_file(config)
    dataset = Toronto3D(cfg.dataset)
    
    tf_data = TF_Dataset(dataset = dataset.get_split('training'), preprocess = kpconv_preprocess)
    loader = tf_data.get_loader()
    # print(loader)
    for data in loader:
        print(data)
        break
        # print("\n\n")
    # loader = SimpleDataset(dataset = dataset.get_split('training'))
    # print(loader)

    # for data in tf_data.spatially_regular_gen():
    #     for a in data:
    #         print(a.shape)
    #     # print(data)
    #     break