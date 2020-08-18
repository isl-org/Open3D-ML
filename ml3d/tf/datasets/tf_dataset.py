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


class TF_Dataset():
    def __init__(self,
                 *args,
                 dataset=None,
                 model=None,
                 no_progress: bool = False,
                 **kwargs):
        self.dataset = dataset
        self.model = model
        self.preprocess = model.preprocess
        self.transform = model.transform
        self.get_batch_gen = model.get_batch_gen
        self.model_cfg = model.cfg

        if self.preprocess is not None:
            cache_dir = getattr(dataset.cfg, 'cache_dir')
            assert cache_dir is not None, 'cache directory is not given'

            self.cache_convert = dataset_helper.Cache(
                self.preprocess,
                cache_dir=cache_dir,
                cache_key=dataset_helper._get_hash(
                    repr(self.preprocess)[:-15]))
            print("cache key : {}".format(repr(self.preprocess)[:-15]))

            uncached = [
                idx for idx in range(len(dataset)) if dataset.get_attr(idx)
                ['name'] not in self.cache_convert.cached_ids
            ]
            if len(uncached) > 0:
                for idx in tqdm(range(len(dataset)),
                                desc='preprocess',
                                disable=no_progress):
                    attr = dataset.get_attr(idx)
                    data = dataset.get_data(idx)
                    name = attr['name']

                    self.cache_convert(name, data, attr)

        else:
            self.cache_convert = None

        self.num_threads = 3  # TODO : read from config
        self.split = dataset.split
        self.pc_list = dataset.path_list
        self.num_pc = len(self.pc_list)

    def read_data(self, key):
        attr = self.dataset.get_attr(key)
        # print(attr)
        if self.cache_convert is None:
            data = self.dataset.get_data(key)
        else:
            data = self.cache_convert(attr['name'])

        return data, attr

    def get_loader(self):
        gen_func, gen_types, gen_shapes = self.get_batch_gen(self)

        tf_dataset = tf.data.Dataset.from_generator(gen_func, gen_types,
                                                    gen_shapes)

        tf_dataset = tf_dataset.map(map_func=self.transform,
                                    num_parallel_calls=self.num_threads)

        return tf_dataset


from ml3d.torch.utils import Config
from ml3d.datasets import Toronto3D

if __name__ == '__main__':
    config = '../../torch/configs/kpconv_toronto3d.py'
    cfg = Config.load_from_file(config)
    dataset = Toronto3D(cfg.dataset)

    tf_data = TF_Dataset(dataset=dataset.get_split('training'),
                         preprocess=kpconv_preprocess)
    loader = tf_data.get_loader()
    # print(loader)
    for data in loader:
        print(data)
        # break
        # print("\n\n")
