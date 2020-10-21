import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
import logging

from .base_dataset import BaseDataset
from ..utils import make_dir, DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class Toronto3D(BaseDataset):
    """
    Toronto3D dataset, used in visualizer, training, or test
    """

    def __init__(self,
                 dataset_path,
                 name='Toronto3D',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[
                     35391894., 1449308., 4650919., 18252779., 589856., 743579.,
                     4311631., 356463.
                 ],
                 ignored_label_inds=[0],
                 train_files=['L001.ply', 'L003.ply', 'L004.ply'],
                 val_files=['L002.ply'],
                 test_files=['L002.ply'],
                 test_result_folder='./test',
                 **kwargs):
        """
        Initialize
        Args:
            dataset_path (str): path to the dataset
            kwargs:
        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         train_files=train_files,
                         test_files=test_files,
                         val_files=val_files,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()

        self.dataset_path = cfg.dataset_path
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(cfg.ignored_label_inds)

        self.train_files = [
            join(self.cfg.dataset_path, f) for f in cfg.train_files
        ]
        self.val_files = [join(self.cfg.dataset_path, f) for f in cfg.val_files]
        self.test_files = [
            join(self.cfg.dataset_path, f) for f in cfg.test_files
        ]

    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: 'Unclassified',
            1: 'Ground',
            2: 'Road_markings',
            3: 'Natural',
            4: 'Building',
            5: 'Utility_line',
            6: 'Pole',
            7: 'Car',
            8: 'Fence'
        }
        return label_to_names

    def get_split(self, split):
        return Toronto3DSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

        return files

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

    def save_test_result(self, results, attr):
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(pred)

        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(path, self.name, name + '.npy')
        make_dir(Path(store_path).parent)
        np.save(store_path, pred)
        log.info("Saved {} in {}.".format(name, store_path))


class Toronto3DSplit():

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

        self.UTM_OFFSET = [627285, 4841948, 0]

        self.cache_in_memory = self.cfg.get('cache_in_memory', False)
        if self.cache_in_memory:
            self.data_list = [None] * len(self.path_list)

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        if self.cache_in_memory:
            if self.data_list[idx] is not None:
                data = self.data_list[idx]
            else:
                data = PlyData.read(pc_path)['vertex']
                self.data_list[idx] = data
        else:
            data = PlyData.read(pc_path)['vertex']

        points = np.vstack(
            (data['x'], data['y'], data['z'])).astype(np.float64).T
        points = points - self.UTM_OFFSET
        points = np.float32(points)

        feat = np.zeros(points.shape, dtype=np.float32)
        feat[:, 0] = data['red']
        feat[:, 1] = data['green']
        feat[:, 2] = data['blue']

        labels = np.array(data['scalar_Label'], dtype=np.int32)

        data = {'point': points, 'feat': feat, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(Toronto3D)
