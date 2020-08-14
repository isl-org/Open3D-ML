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
                input_labels = data['label'][input_inds]
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

        tf_dataset = tf.data.Dataset.from_generator(self.spatially_regular_gen, gen_shapes)
        # tf_dataset = tf.data.Dataset.range(len(self.dataset))
        # tf_dataset = tf_dataset.map(lambda x : tf.numpy_function(func = self.read_data, inp = [x], Tout = [tf.float32, tf.float32,
        #                             tf.int32]))

        # tf_dataset = tf_dataset.map(map_func = self.transform)

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
    print(loader)
    for data in loader:
        print(data)
        break
        print("\n\n")
    # loader = SimpleDataset(dataset = dataset.get_split('training'))
    # print(loader)
