import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
import logging

from .utils import DataProcessing as DP
from .base_dataset import BaseDataset
from ..utils import make_dir, DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class Semantic3D(BaseDataset):
    """
    SemanticKITTI dataset, used in visualizer, training, or test
    """

    def __init__(self,
                 dataset_path,
                 name='Semantic3D',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[
                     5181602, 5012952, 6830086, 1311528, 10476365, 946982,
                     334860, 269353
                 ],
                 ignored_label_inds=[0],
                 val_files=[
                     'bildstein_station3_xyz_intensity_rgb',
                     'sg27_station2_intensity_rgb'
                 ],
                 test_result_folder='./test',
                 **kwargs):
        """
        Initialize
        Args:
            dataset_path: path to the dataset
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
                         val_files=val_files,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        self.all_files = glob.glob(str(Path(self.cfg.dataset_path) / '*.txt'))

        self.train_files = [
            f for f in self.all_files if exists(
                str(Path(f).parent / Path(f).name.replace('.txt', '.labels')))
        ]
        self.test_files = [
            f for f in self.all_files if f not in self.train_files
        ]

        self.train_files = np.sort(self.train_files)
        self.test_files = np.sort(self.test_files)
        self.val_files = []

        for i, file_path in enumerate(self.train_files):
            for val_file in cfg.val_files:
                if val_file in file_path:
                    self.val_files.append(file_path)
                    break

        self.train_files = np.sort(
            [f for f in self.train_files if f not in self.val_files])

    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: 'unlabeled',
            1: 'man-made terrain',
            2: 'natural terrain',
            3: 'high vegetation',
            4: 'low vegetation',
            5: 'buildings',
            6: 'hard scape',
            7: 'scanning artefacts',
            8: 'cars'
        }
        return label_to_names

    def get_split(self, split):
        return Semantic3DSplit(self, split=split)

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
        store_path = join(path, self.name, name + '.labels')
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

        pred = results['predict_labels'] + 1
        store_path = join(path, self.name, name + '.labels')
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info("Saved {} in {}.".format(name, store_path))


class Semantic3DSplit():

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        pc = pd.read_csv(pc_path,
                         header=None,
                         delim_whitespace=True,
                         dtype=np.float32).values

        points = pc[:, 0:3]
        feat = pc[:, [4, 5, 6]]
        intensity = pc[:, 3]

        points = np.array(points, dtype=np.float32)
        feat = np.array(feat, dtype=np.float32)
        intensity = np.array(intensity, dtype=np.float32)

        if (self.split != 'test'):
            labels = pd.read_csv(pc_path.replace(".txt", ".labels"),
                                 header=None,
                                 delim_whitespace=True,
                                 dtype=np.int32).values
            labels = np.array(labels, dtype=np.int32).reshape((-1,))
        else:
            labels = np.zeros((points.shape[0],), dtype=np.int32)

        data = {
            'point': points,
            'feat': feat,
            'intensity': intensity,
            'label': labels
        }

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(Semantic3D)
