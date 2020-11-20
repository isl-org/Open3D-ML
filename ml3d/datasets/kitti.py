import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
import logging
import yaml

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import Config, make_dir, DATASET
from ..vis.boundingbox import BoundingBox3D

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class KITTI(BaseDataset):
    """
    KITTI 3D dataset for Object Detection, used in visualizer, training, or test
    """

    def __init__(self,
                 dataset_path,
                 name='KITTI',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 val_split=3712,
                 **kwargs):
        """
        Initialize
        Args:
            dataset_path (str): path to the dataset
            kwargs:
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         val_split=val_split,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 3
        self.label_to_names = self.get_label_to_names()

        self.all_files = glob(
            join(cfg.dataset_path, 'training', 'velodyne', '*.bin'))
        self.train_files = []
        self.val_files = []

        for f in self.all_files:
            idx = int(Path(f).name.replace('.bin', ''))
            if idx < cfg.val_split:
                self.train_files.append(f)
            else:
                self.val_files.append(f)

        self.test_files = glob(
            join(cfg.dataset_path, 'testing', 'velodyne', '*.bin'))

    @staticmethod
    def get_label_to_names():
        label_to_names = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Van'}
        return label_to_names

    @staticmethod
    def read_lidar(path):
        assert Path(path).exists()

        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def read_label(path, calib):
        if not Path(path).exists():
            return None

        with open(path, 'r') as f:
            lines = f.readlines()

        objects = []
        for line in lines:
            label = line.strip().split(' ')

            center = np.array(
                [float(label[11]),
                 float(label[12]),
                 float(label[13]), 1.0])

            rect = calib['R0_rect']
            Trv2c = calib['Tr_velo2cam']

            points = center @ np.linalg.inv((rect @ Trv2c).T)

            size = [float(label[9]), float(label[8]), float(label[10])]  # w,h,l
            center = [points[0], points[1], size[1] / 2 + points[2]]

            ry = float(label[14])
            front = [-1 * np.sin(ry), -1 * np.cos(ry), 0]
            up = [0, 0, 1]
            left = [-1 * np.cos(ry), np.sin(ry), 0]

            objects.append(Object3d(center, front, up, left, size, label))

        return objects

    @staticmethod
    def _extend_matrix(mat):
        mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
        return mat

    @staticmethod
    def read_calib(path):
        assert Path(path).exists()

        with open(path, 'r') as f:
            lines = f.readlines()

        obj = lines[0].strip().split(' ')[1:]
        P0 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[1].strip().split(' ')[1:]
        P1 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32).reshape(3, 4)

        P0 = KITTI._extend_matrix(P0)
        P1 = KITTI._extend_matrix(P1)
        P2 = KITTI._extend_matrix(P2)
        P3 = KITTI._extend_matrix(P3)

        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32).reshape(3, 3)

        rect_4x4 = np.zeros([4, 4], dtype=R0.dtype)
        rect_4x4[3, 3] = 1
        rect_4x4[:3, :3] = R0

        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32).reshape(3, 4)
        Tr_velo_to_cam = KITTI._extend_matrix(Tr_velo_to_cam)

        return {
            'P0': P0,
            'P1': P1,
            'P2': P2,
            'P3': P3,
            'R0_rect': rect_4x4,
            'Tr_velo2cam': Tr_velo_to_cam
        }

    def get_split(self, split):
        return KITTISplit(self, split=split)

    def get_split_list(self, split):
        cfg = self.cfg
        dataset_path = cfg.dataset_path
        file_list = []

        if split in ['train', 'training']:
            return self.train_files
            seq_list = cfg.training_split
        elif split in ['test', 'testing']:
            return self.test_files
        elif split in ['val', 'validation']:
            return val_files
        elif split in ['all']:
            return self.train_files + self.val_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested():
        pass

    def save_test_result():
        pass


class KITTISplit():

    def __init__(self, dataset, split='train'):
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
        label_path = pc_path.replace('velodyne',
                                     'label_2').replace('.bin', '.txt')
        calib_path = label_path.replace('label_2', 'calib')

        pc = self.dataset.read_lidar(pc_path)
        calib = self.dataset.read_calib(calib_path)
        label = self.dataset.read_label(label_path, calib)

        data = {
            'point': pc,
            'feat': None,
            'calib': calib,
            'bounding_boxes': label,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr


class Object3d(BoundingBox3D):
    """
    Stores object specific details like bbox coordinates, occlusion etc.
    """

    def __init__(self, center, front, up, left, size, label):

        label_class = self.cls_type_to_id(label[0])
        confidence = float(label[15]) if label.__len__() == 16 else -1.0

        super().__init__(center, front, up, left, size, label_class, confidence)

        self.name = label[0]
        self.cls_id = self.cls_type_to_id(self.name)
        self.truncation = float(label[1])
        self.occlusion = float(
            label[2]
        )  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown

        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(
            label[6]), float(label[7])),
                              dtype=np.float32)

        self.dis_to_cam = np.linalg.norm(self.center)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level = self.get_kitti_obj_level()

    @staticmethod
    def cls_type_to_id(cls_type):
        """
        get object id from name.
        """
        type_to_id = {
            'DontCare': 0,
            'Car': 1,
            'Pedestrian': 2,
            'Cyclist': 3,
            'Van': 4
        }
        if cls_type not in type_to_id.keys():
            return 0
        return type_to_id[cls_type]

    def get_kitti_obj_level(self):
        """
        determines the difficulty level of object.
        """
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.size[2::-1]
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)], [0, 1, 0],
                      [-np.sin(self.ry), 0,
                       np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.center
        return corners3d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.name, self.truncation, self.occlusion, self.alpha, self.box2d, self.size[2], self.size[0], self.size[1],
                        self.center, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.name, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.size[2], self.size[0], self.size[1], self.center[0], self.center[1], self.center[2],
                       self.ry)
        return kitti_str


DATASET._register_module(KITTI)
