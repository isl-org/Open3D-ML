import numpy as np
import pandas as pd
import os, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath, isdir
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging

from .utils import DataProcessing, get_min_bbox, BEVBox3D
from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET, Config, get_module

log = logging.getLogger(__name__)


class MegaLoader():
    """This class is used to create a combination of multiple datasets,
    and sample data among them uniformly.
    """

    def __init__(self,
                 config_paths,
                 batch_size=1,
                 name='MegaLoader',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 ignored_label_inds=[],
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            config_paths: List of dataset config files to use.
            dataset_path: The path to the dataset to use (parent directory of data_3d_semantics).
            name: The name of the dataset (MegaLoader in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.
        """

        kwargs['config_paths'] = config_paths
        kwargs['name'] = name
        kwargs['cache_dir'] = cache_dir
        kwargs['use_cache'] = use_cache
        kwargs['ignored_label_inds'] = ignored_label_inds
        kwargs['test_result_folder'] = test_result_folder
        kwargs['batch_size'] = batch_size

        self.cfg = Config(kwargs)
        self.name = self.cfg.name
        self.batch_size = batch_size
        self.rng = np.random.default_rng(kwargs.get('seed', None))
        self.ignored_labels = np.array([])

        self.num_datasets = len(config_paths)

        self.configs = [
            Config.load_from_file(cfg_path) for cfg_path in config_paths
        ]
        self.datasets = [
            get_module('dataset', cfg.name)(**cfg) for cfg in self.configs
        ]
        self.class_weights = [self.datasets[i].cfg.class_weights for i in range(self.num_datasets)]


    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return MegaLoaderSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['train', 'training']:
            return self.train_files
        elif split in ['val', 'validation']:
            return self.val_files
        elif split in ['test', 'testing']:
            return test_files
        elif split == 'all':
            return self.train_files + self.val_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, name):
        """Saves the output of a model.

            Args:
                results: The output of a model for the datum associated with the attribute passed.
                attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(pred)

        store_path = join(path, self.name, name + '.npy')
        make_dir(Path(store_path).parent)
        np.save(store_path, pred)
        log.info("Saved {} in {}.".format(name, store_path))


class MegaLoaderSplit():
    """This class is used to create a split for MegaLoader dataset.

    Initialize the class.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training', test_dataset_idx=0):
        self.cfg = dataset.cfg
        self.split = split
        self.dataset = dataset
        self.dataset_splits = [
            a.get_split(split) for a in self.dataset.datasets
        ]
        self.num_datasets = dataset.num_datasets
        self.test_dataset_idx = test_dataset_idx

        if 'test' in split:
            sampler_cls = get_module('sampler', 'SemSegSpatiallyRegularSampler')
            self.sampler = sampler_cls(self)

    def __len__(self):
        if 'test' in self.split:
            return len(self.dataset_splits[self.test_dataset_idx])
        lens = [len(a) for a in self.dataset_splits]
        return max(lens) * self.num_datasets

    def get_data(self, idx):
        if 'test' in self.split:
            dataset_idx = self.test_dataset_idx
        else:
            dataset_idx = idx % self.num_datasets
            idx = (idx // self.num_datasets) % len(self.dataset_splits[dataset_idx])

        data = self.dataset_splits[dataset_idx].get_data(idx)

        return data

    def get_attr(self, idx):
        if 'test' in self.split:
            dataset_idx = self.test_dataset_idx
        else:
            dataset_idx = idx % self.num_datasets
            idx = (idx // self.num_datasets) % len(self.dataset_splits[dataset_idx])

        attr = self.dataset_splits[dataset_idx].get_attr(idx)
        attr['dataset_idx'] = dataset_idx

        return attr


DATASET._register_module(MegaLoader)
