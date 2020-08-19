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

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)

# Expect point clouds to be in npy format with train, val and test files in separate folders.
# Expected format of npy files : ['x', 'y', 'z', 'class', 'feat_1', 'feat_2', ........,'feat_n'].
# For test files, format should be : ['x', 'y', 'z', 'feat_1', 'feat_2', ........,'feat_n'].

class Custom3DSplit():
    def __init__(self, dataset, split='training'):
        self.cfg    = dataset.cfg
        path_list   = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        data = np.load(pc_path)
        points = np.array(data[:, :3], dtype = np.float32)
        
        if(self.split != 'test'):
            labels = np.array(data[:, 3], dtype = np.int32)
            feat = data[:, 4:] if data.shape[1] > 4 else None
        else:
            feat = np.array(data[:, 3:], dtype = np.float32) if data.shape[1] > 3 else None
            labels = np.zeros((points.shape[0], ), dtype = np.int32)
        
        data = {
            'point' : points,
            'feat' : feat,
            'label' : labels
        }

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.npy', '')

        attr = {
            'name'      : name,
            'path'      : str(pc_path),
            'split'     : self.split
        }
        return attr


class Custom3D:
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = 'Custom3D'
        self.dataset_path = cfg.dataset_path
        self.label_to_names = {0: 'Unclassified',
                               1: 'Ground',
                               2: 'Road_markings',
                               3: 'Natural',
                               4: 'Building',
                               5: 'Utility_line',
                               6: 'Pole',
                               7: 'Car',
                               8: 'Fence'}

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        self.train_dir = str(Path(cfg.dataset_path) / cfg.train_dir)
        self.val_dir = str(Path(cfg.dataset_path) / cfg.val_dir)
        self.test_dir = str(Path(cfg.dataset_path) / cfg.test_dir)

        self.train_files = [f for f in glob.glob(self.train_dir + "/*.npy")]
        self.val_files = [f for f in glob.glob(self.val_dir + "/*.npy")]
        self.test_files = [f for f in glob.glob(self.test_dir + "/*.npy")]

    def get_split (self, split):
        return Custom3DSplit(self, split=split)
    
    def get_split_list(self, split):
        if split == 'test':
            random.shuffle(self.test_files)
            return self.test_files
        elif split == 'val':
            random.shuffle(self.val_files)
            return self.val_files
        else:
            random.shuffle(self.train_files)
            return self.train_files



from ml3d.utils import Config


if __name__ == '__main__':
    config = '../configs/randlanet_custom3d.py'
    cfg  = Config.load_from_file(config)
    a = Custom3D(cfg.dataset)
    b = a.get_split("test")
    # c = b.get_data(1)
    print(b.get_attr(0))
    # print(b.get_attr(1)['name'])
    # print(c['point'].shape)
    # print(c['feat'].shape)
    # print(c['label'].shape)
    # print(c['point'][0])
    # print(c['feat'][0])
    # print(c['label'][0])
    # print(c['label'].mean())
    # print(c['label'])
    # # print(b.get_data(0))
    # print(b.get_attr(10))