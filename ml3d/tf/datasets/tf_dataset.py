from abc import abstractmethod
from tqdm import tqdm
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
import random

import tensorflow as tf
import numpy as np
from ml3d.torch.utils import dataset_helper

from ml3d.datasets.utils import DataProcessing
from sklearn.neighbors import KDTree

# Load custom operation
BASE_DIR = Path(abspath(__file__))

tf_neighbors_module = tf.load_op_library(str(BASE_DIR.parent.parent / 'utils' / 'tf_custom_ops' / 'tf_neighbors.so'))
tf_batch_neighbors_module = tf.load_op_library(str(BASE_DIR.parent.parent / 'utils' / 'tf_custom_ops' / 'tf_batch_neighbors.so'))
tf_subsampling_module = tf.load_op_library(str(BASE_DIR.parent.parent / 'utils' / 'tf_custom_ops' / 'tf_subsampling.so'))
tf_batch_subsampling_module = tf.load_op_library(str(BASE_DIR.parent.parent / 'utils' / 'tf_custom_ops' / 'tf_batch_subsampling.so'))

def tf_batch_subsampling(points, batches_len, sampleDl):
    return tf_batch_subsampling_module.batch_grid_subsampling(points, batches_len, sampleDl)

def tf_batch_neighbors(queries, supports, q_batches, s_batches, radius):
    return tf_batch_neighbors_module.batch_ordered_neighbors(queries, supports, q_batches, s_batches, radius)


# def randlanet_preprocess():

class TF_Dataset():
    def __init__(self,
                 *args,
                 dataset=None,
                 preprocess=None,
                 transform=None,
                 generator = None,
                 cfg = None,
                 no_progress: bool = False,
                 **kwargs):
        self.dataset = dataset
        self.preprocess = preprocess
        self.transform = transform
        self.get_batch_gen = generator
        self.model_cfg = cfg

        if preprocess is not None:
            cache_dir = getattr(dataset.cfg, 'cache_dir')
            assert cache_dir is not None, 'cache directory is not given'

            self.cache_convert = dataset_helper.Cache(
                preprocess,
                cache_dir=cache_dir,
                cache_key=dataset_helper._get_hash(repr(preprocess)[:-15]))
            print("cache key : {}".format(repr(preprocess)[:-15]))

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

        # pick_idx = np.random.choice(len(data['point']), 1)
        # pc, feat, label, _ = crop_pc(data['point'], data['feat'], data['label'], data['search_tree'], pick_idx)

        return data, attr
        # return pc, feat, label



    def get_loader(self):

        gen_func, gen_types, gen_shapes = self.get_batch_gen(self, self.model_cfg)

        tf_dataset = tf.data.Dataset.from_generator(gen_func, gen_types, gen_shapes)

        # tf_dataset = tf.data.Dataset.range(len(self.dataset))
        # tf_dataset = tf_dataset.map(lambda x : tf.numpy_function(func = self.read_data, inp = [x], Tout = [tf.float32, tf.float32,
        #                             tf.int32]))

        # tf_dataset = tf_dataset.map(map_func = self.transform)
        tf_dataset = tf_dataset.map(map_func=self.transform, num_parallel_calls=self.num_threads)


        return tf_dataset

    # def big_neighborhood_filter(self, neighbors, layer):
    #     """
    #     Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
    #     Limit is computed at initialization
    #     """
    #     # crop neighbors matrix
    #     return neighbors[:, :neighborhood_limits[layer]]


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
    config = '../../torch/configs/kpconv_toronto3d.py'
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