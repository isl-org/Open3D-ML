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

from .base_dataset import BaseDataset
from ..utils import make_dir, DATASET


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class ParisLille3D(BaseDataset):
    """
    ParisLille3D dataset, used in visualizer, training, or test
    """
    def __init__(self, 
                name='ParisLille3D',
                cache_dir='./logs/cache', 
                use_cache=False,  
                num_points=65536,
                class_weights=[
                    5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353,
                    269353
                ],
                dataset_path='../dataset/Paris_Lille3D/',
                test_result_folder='./test',
                val_files=['Lille2.ply']
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
        super().__init__(
                        name=name,
                        cache_dir=cache_dir, 
                        use_cache=use_cache, 
                        class_weights=class_weights,
                        num_points=num_points,
                        dataset_path=dataset_path, 
                        test_result_folder=test_result_folder,
                        val_files=val_files)

        cfg = self.cfg

        self.label_to_names = {
            0: 'unclassified',
            1: 'ground',
            2: 'building',
            3: 'pole-road_sign-traffic_light',
            4: 'bollard-small_pole',
            5: 'trash_can',
            6: 'barrier',
            7: 'pedestrian',
            8: 'car',
            9: 'natural-vegetation'
        }

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort(
            [k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        train_path = cfg.dataset_path + "/training_10_classes/"
        self.train_files = glob.glob(train_path + "/*.ply")
        self.val_files = [
            f for f in self.train_files if Path(f).name in cfg.val_files
        ]
        self.train_files = [
            f for f in self.train_files if f not in self.val_files
        ]

        test_path = cfg.dataset_path + "/test_10_classes/"
        self.test_files = glob.glob(test_path + '*.ply')

    def get_split(self, split):
        return ParisLille3DSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['test', 'testing']:
            random.shuffle(self.test_files)
            return self.test_files
        elif split in ['val', 'validation']:
            random.shuffle(self.val_files)
            return self.val_files
        elif split in ['train', 'training']:
            random.shuffle(self.train_files)
            return self.train_files
        else:
            raise ValueError("Invalid split {}".format(split))


class ParisLille3DSplit():
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
        data = PlyData.read(pc_path)['vertex']

        points = np.zeros((data['x'].shape[0], 3), dtype=np.float32)
        points[:, 0] = data['x']
        points[:, 1] = data['y']
        points[:, 2] = data['z']

        if (self.split != 'test'):
            labels = np.array(data['class'], dtype=np.int32)
        else:
            labels = np.zeros((points.shape[0], ), dtype=np.int32)

        data = {'point': points, 'feat': None, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.ply', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr

DATASET._register_module(ParisLille3D)

