import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from tqdm import tqdm
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging

from ..utils import make_dir, DATASET
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
                name='Toronto3D',
                cache_dir='./logs/cache',
                dataset_path='../dataset/Semantic3D/', 
                use_cache=False,  
                num_points=65536,
                prepro_grid_size=0.06,
                class_weights=[
                    5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353
                ],
                ignored_label_inds=[0],
                val_split=1,
                ):
        """
        Initialize
        Args:
            cfg (cfg object or str): cfg object or path to cfg file
            dataset_path (str): path to the dataset
            args (dict): dict of args 
            kwargs:
        Returns:
            class: The corresponding class.
        """
        super().__init__(name=name,
                        cache_dir=cache_dir, 
                        use_cache=use_cache, 
                        class_weights=class_weights,
                        num_points=num_points,
                        prepro_grid_size=prepro_grid_size,
                        dataset_path=dataset_path, 
                        ignored_label_inds=ignored_label_inds,
                        val_split=val_split)

        cfg = self.cfg

        self.label_to_names = {
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
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort(
            [k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        self.all_files = glob.glob(str(Path(self.cfg.dataset_path) / '*.txt'))
        random.shuffle(self.all_files)

        for f in self.all_files:
            print(Path(f).name.replace('.txt', '.labels'))
            print(Path(f).parent)
            print(Path(f).parent / Path(f).name.replace('.txt', '.labels'))
            print("=====")

        self.train_files = [
            f for f in self.all_files if exists(
                str(Path(f).parent / Path(f).name.replace('.txt', '.labels')))
        ]
        self.test_files = [
            f for f in self.all_files if f not in self.train_files
        ]

        self.all_split = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.val_split = cfg.val_split

        self.train_files = np.sort(self.train_files)
        self.test_files = np.sort(self.test_files)
        self.val_files = []

        for i, file_path in enumerate(self.train_files):
            if self.all_split[i] == self.val_split:
                self.val_files.append(file_path)

        self.train_files = np.sort(
            [f for f in self.train_files if f not in self.val_files])

    def get_split(self, split):
        return Semantic3DSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['test', 'testing']:
            random.shuffle(self.test_files)
            return self.test_files
        elif split in ['train', 'training']:
            random.shuffle(self.train_files)
            return self.train_files
        elif split in ['val', 'validation']:
            random.shuffle(self.val_files)
            return self.val_files
        else:
            raise ValueError("Invalid split {}".format(split))


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
        feat = pc[:, [4, 5, 6, 3]]

        points = np.array(points, dtype=np.float32)
        feat = np.array(feat, dtype=np.float32)

        if (self.split != 'test'):
            labels = pd.read_csv(pc_path.replace(".txt", ".labels"),
                                 header=None,
                                 delim_whitespace=True,
                                 dtype=np.int32).values
            labels = np.array(labels, dtype=np.int32)
        else:
            labels = np.zeros((points.shape[0], ), dtype=np.int32)

        data = {'point': points, 'feat': feat, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr

DATASET._register_module(Semantic3D)
