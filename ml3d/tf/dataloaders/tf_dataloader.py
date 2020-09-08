from abc import abstractmethod
from tqdm import tqdm
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
import random

import tensorflow as tf
import numpy as np
from ...utils import dataset_helper

from ...datasets.utils import DataProcessing
from sklearn.neighbors import KDTree


class TFDataloader():
    def __init__(self,
                 *args,
                 dataset=None,
                 model=None,
                 use_cache=True,
                 **kwargs):
        self.dataset = dataset
        self.model = model
        self.preprocess = model.preprocess
        self.transform = model.transform
        self.get_batch_gen = model.get_batch_gen
        self.model_cfg = model.cfg

        if self.preprocess is not None and use_cache:
            cache_dir = getattr(dataset.cfg, 'cache_dir')

            assert cache_dir is not None, 'cache directory is not given'

            self.cache_convert = dataset_helper.Cache(
                self.preprocess,
                cache_dir=cache_dir,
                cache_key=dataset_helper._get_hash(
                    repr(self.preprocess)[:-15]))

            uncached = [
                idx for idx in range(len(dataset)) if dataset.get_attr(idx)
                ['name'] not in self.cache_convert.cached_ids
            ]
            if len(uncached) > 0:
                print("cache key : {}".format(repr(self.preprocess)[:-15]))
                for idx in tqdm(range(len(dataset)),
                                desc='preprocess'):
                    attr = dataset.get_attr(idx)
                    data = dataset.get_data(idx)
                    name = attr['name']

                    self.cache_convert(name, data, attr)

        else:
            self.cache_convert = None
        self.split = dataset.split
        self.pc_list = dataset.path_list
        self.num_pc = len(self.pc_list)

    def read_data(self, index):
        attr = self.dataset.get_attr(index)
        if self.cache_convert:
            data = self.cache_convert(attr['name'])
        elif self.preprocess:
            data = self.preprocess(self.dataset.get_data(index), attr)
        else:
            data = self.dataset.get_data(index)


        return data, attr

    def get_loader(self, batch_size=1, num_threads=3):
        gen_func, gen_types, gen_shapes = self.get_batch_gen(self)

        loader = tf.data.Dataset.from_generator(gen_func, gen_types,
                                                gen_shapes)

        loader = loader.map(map_func=self.transform,
                            num_parallel_calls=num_threads)

        print(self.model_cfg.batcher)
        print(vars(self.model_cfg))
        if ( self.model_cfg.__getattr__('batcher') is not None
                or self.model_cfg.batcher == 'DefaultBatcher'):
            loader = loader.batch(batch_size)

        return loader
