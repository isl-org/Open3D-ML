import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
import logging
import yaml

from .base_dataset import BaseDataset
from ..utils import Config, make_dir, DATASET
from ..vis.boundingbox import BoundingBox3D

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class Lyft(BaseDataset):
    """
    Lyft level 5 dataset for Object Detection, used in visualizer, training, or test
    """

    def __init__(self,
                 dataset_path,
                 info_path,
                 name='Lyft',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 **kwargs):
        """
        Initialize
        Args:
            dataset_path (str): path to the dataset
            kwargs:
        """
        super().__init__(dataset_path=dataset_path,
                         info_path=info_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 9
        self.label_to_names = self.get_label_to_names()

        self.train_info = {}
        self.test_info = {}
        self.val_info = {}

        if os.path.exists(join(info_path, 'infos_train.pkl')):
            self.train_info = pickle.load(
                open(join(info_path, 'infos_train.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'infos_val.pkl')):
            self.val_info = pickle.load(
                open(join(info_path, 'infos_val.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'infos_test.pkl')):
            self.test_info = pickle.load(
                open(join(info_path, 'infos_test.pkl'), 'rb'))

    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: 'ignore',
            1: 'bicycle',
            2: 'bus',
            3: 'car',
            4: 'emergency_vehicle',
            5: 'motorcycle',
            6: 'other_vehicle',
            7: 'pedestrian',
            8: 'truck',
            9: 'animal'
        }

        return label_to_names

    @staticmethod
    def read_lidar(path):
        assert Path(path).exists()

        return np.fromfile(path, dtype=np.float32).reshape(-1, 5)

    @staticmethod
    def read_label(info):
        mask = info['num_lidar_pts'] != 0
        boxes = info['gt_boxes'][mask]
        names = info['gt_names'][mask]

        objects = []
        for name, box in zip(names, boxes):
            center = [float(box[0]), float(box[1]), float(box[2])]
            size = [float(box[3]), float(box[5]), float(box[4])]
            ry = float(box[6])
            front = [-1 * np.sin(ry), -1 * np.cos(ry), 0]
            up = [0, 0, 1]
            left = [-1 * np.cos(ry), np.sin(ry), 0]

            objects.append(Object3d(center, front, up, left, size, name, box))

        return objects

    def get_split(self, split):
        return LyftSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['train', 'training']:
            return self.train_info
        elif split in ['test', 'testing']:
            return self.test_info
        elif split in ['val', 'validation']:
            return self.val_info

        raise ValueError("Invalid split {}".format(split))

    def is_tested():
        pass

    def save_test_result():
        pass


class LyftSplit():

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg

        self.infos = dataset.get_split_list(split)

        log.info("Found {} pointclouds for {}".format(len(self.infos), split))

        self.path_list = []
        for info in self.infos:
            self.path_list.append(info['lidar_path'])
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.infos)

    def get_data(self, idx):
        info = self.infos[idx]
        lidar_path = info['lidar_path']

        pc = self.dataset.read_lidar(lidar_path)
        label = self.dataset.read_label(info)

        calib = {
            'lidar2ego_tr': info['lidar2ego_tr'],
            'lidar2ego_rot': info['lidar2ego_rot'],
            'ego2global_tr': info['ego2global_tr'],
            'ego2global_rot': info['ego2global_rot'],
        }

        data = {
            'point': pc,
            'feat': None,
            'calib': calib,
            'bounding_boxes': label,
        }

        return data

    def get_attr(self, idx):
        info = self.infos[idx]
        pc_path = info['lidar_path']
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


class Object3d(BoundingBox3D):
    """
    Stores object specific details like bbox coordinates.
    """

    def __init__(self, center, front, up, left, size, name, box):
        label_class = self.cls_type_to_id(name)

        super().__init__(center, front, up, left, size, label_class, 1.0)

        self.name = name
        self.cls_id = self.cls_type_to_id(name)
        self.dis_to_cam = np.linalg.norm(self.center)
        self.ry = float(box[6])

    @staticmethod
    def cls_type_to_id(cls_type):
        """
        get object id from name.
        """
        type_to_id = {
            'ignore': 0,
            'bicycle': 1,
            'bus': 2,
            'car': 3,
            'emergency_vehicle': 4,
            'motorcycle': 5,
            'other_vehicle': 6,
            'pedestrian': 7,
            'truck': 8,
            'animal': 9,
        }
        if cls_type not in type_to_id.keys():
            return -1
        return type_to_id[cls_type]

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)], [0, 1, 0],
                      [-np.sin(self.ry), 0,
                       np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.loc
        return corners3d


DATASET._register_module(Lyft)
