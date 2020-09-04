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

from .base_dataset import BaseDataset
from ..utils import make_dir, DATASET


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
                name='Toronto3D',
                cache_dir='./logs/cache', 
                use_cache=False,  
                num_points=65536,
                prepro_grid_size=0.06,
                class_weights=[
                    5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353
                ],
                ignored_label_inds=[0],
                dataset_path='../dataset/Toronto_3D/',
                train_files=['L001.ply', 'L003.ply', 'L004.ply'],
                val_files=['L002.ply'],
                test_files=['L002.ply'],
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
                        prepro_grid_size=prepro_grid_size,
                        dataset_path=dataset_path, 
                        ignored_label_inds=ignored_label_inds,
                        train_files=train_files,
                        test_files=test_files,
                        val_files=val_files)

        cfg = self.cfg

        self.label_to_names = {
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

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort(
            [k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        self.train_files = [join(self.cfg.dataset_path, f) for f in cfg.train_files]
        self.val_files = [join(self.cfg.dataset_path, f) for f in cfg.val_files]
        self.test_files = [join(self.cfg.dataset_path, f) for f in cfg.test_files]


    def get_split(self, split):
        return Toronto3DSplit(self, split=split)

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


class Toronto3DSplit():
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

        feat = np.zeros(points.shape, dtype=np.float32)
        feat[:, 0] = data['red']
        feat[:, 1] = data['green']
        feat[:, 2] = data['blue']

        if (self.split != 'test'):
            labels = np.array(data['scalar_Label'], dtype=np.int32)
        else:
            labels = np.zeros((points.shape[0], ), dtype=np.int32)

        data = {'point': points, 'feat': feat, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr

DATASET._register_module(Toronto3D)
