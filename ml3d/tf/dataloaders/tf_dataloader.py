from abc import abstractmethod
from tqdm import tqdm
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
import random

import tensorflow as tf
import numpy as np
from ...utils import Cache, get_hash

from ...datasets.utils import DataProcessing
from sklearn.neighbors import KDTree


class TFDataloader():
    """
    Data loader for tf framework.
    """

    def __init__(self,
                 *args,
                 dataset=None,
                 model=None,
                 use_cache=True,
                 steps_per_epoch=None,
                 **kwargs):
        """
        Initialize

        Args:
            dataset: ml3d dataset class.
            dataset: model's preprocess method.
            devce: model's transform mthod.
            use_cache: whether to use cached preprocessed data.
            steps_per_epoch: steps per epoch. The step number will be the
                number of samples in the data if steps_per_epoch=None
            kwargs:
        Returns:
            class: The corresponding class.
        """
        self.dataset = dataset
        self.model = model
        self.preprocess = model.preprocess
        self.transform = model.transform
        self.get_batch_gen = model.get_batch_gen
        self.model_cfg = model.cfg
        self.steps_per_epoch = steps_per_epoch

        if self.preprocess is not None and use_cache:
            cache_dir = getattr(dataset.cfg, 'cache_dir')

            assert cache_dir is not None, 'cache directory is not given'

            self.cache_convert = Cache(self.preprocess,
                                       cache_dir=cache_dir,
                                       cache_key=get_hash(
                                           repr(self.preprocess)[:-15]))

            uncached = [
                idx for idx in range(len(dataset)) if dataset.get_attr(idx)
                ['name'] not in self.cache_convert.cached_ids
            ]
            if len(uncached) > 0:
                print("cache key : {}".format(repr(self.preprocess)[:-15]))
                for idx in tqdm(range(len(dataset)), desc='preprocess'):
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
        """Returns the data at index idx. """
        attr = self.dataset.get_attr(index)
        if self.cache_convert:
            data = self.cache_convert(attr['name'])
        elif self.preprocess:
            data = self.preprocess(self.dataset.get_data(index), attr)
        else:
            data = self.dataset.get_data(index)

        return data, attr

    def get_loader(self, batch_size=1, num_threads=3):
        """
        Construct the origianl tensorflow dataloader.

        Args:
            batch_size: batch size.
            num_threads: number of threads for data loading.
            kwargs:
        Returns:
            the tensorflow dataloader and the number of steps in one epoch
        """

        gen_func, gen_types, gen_shapes = self.get_batch_gen(
            self, self.steps_per_epoch, batch_size)

        loader = tf.data.Dataset.from_generator(gen_func, gen_types, gen_shapes)

        loader = loader.map(map_func=self.transform,
                            num_parallel_calls=num_threads)

        if ('batcher' not in self.model_cfg.keys() or
                self.model_cfg.batcher == 'DefaultBatcher'):
            loader = loader.batch(batch_size)

        length = len(self.dataset) / batch_size + 1 if len(
            self.dataset) % batch_size else len(self.dataset) / batch_size
        length = length if self.steps_per_epoch is None else self.steps_per_epoch

        return loader, int(length)
