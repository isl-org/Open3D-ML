import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
import logging
import yaml

from .base_dataset import BaseDataset
from ..utils import Config, make_dir, DATASET
from .utils import BEVBox3D

log = logging.getLogger(__name__)


class Waymo(BaseDataset):
    """This class is used to create a dataset based on the Waymo 3D dataset, and
    used in object detection, visualizer, training, or testing.

    The Waymo 3D dataset is best suited for autonomous driving applications.
    """

    def __init__(self,
                 dataset_path,
                 name='Waymo',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Waymo in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 4
        self.label_to_names = self.get_label_to_names()
        self.shuffle = kwargs.get('shuffle', False)

        self.all_files = sorted(
            glob(join(cfg.dataset_path, 'velodyne', '*.bin')))
        self.train_files = []
        self.val_files = []
        self.test_files = []

        for f in self.all_files:
            if 'train' in f:
                self.train_files.append(f)
            elif 'val' in f:
                self.val_files.append(f)
            elif 'test' in f:
                self.test_files.append(f)
            else:
                log.warning(
                    f"Skipping {f}, prefix must be one of train, test or val.")
        if self.shuffle:
            log.info("Shuffling training files...")
            self.rng.shuffle(self.train_files)

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'PEDESTRIAN',
            1: 'VEHICLE',
            2: 'CYCLIST',
            3: 'SIGN'
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            pc: pointcloud data with shape [N, 6], where
                the format is xyzRGB.
        """
        return np.fromfile(path, dtype=np.float32).reshape(-1, 6)

    @staticmethod
    def read_label(path, calib):
        """Reads labels of bounding boxes.

        Args:
            path: The path to the label file.
            calib: Calibration as returned by read_calib().

        Returns:
            The data objects with bounding boxes information.
        """
        if not Path(path).exists():
            return None

        with open(path, 'r') as f:
            lines = f.readlines()

        objects = []
        for line in lines:
            label = line.strip().split(' ')
            center = [float(label[11]), float(label[12]), float(label[13])]
            size = [float(label[9]), float(label[8]), float(label[10])]
            objects.append(Object3d(center, size, label, calib))

        return objects

    @staticmethod
    def _extend_matrix(mat):
        mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
        return mat

    @staticmethod
    def read_calib(path):
        """Reads calibiration for the dataset. You can use them to compare
        modeled results to observed results.

        Returns:
            The camera and the camera image used in calibration.
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        obj = lines[0].strip().split(' ')[1:]
        unused_P0 = np.array(obj, dtype=np.float32)

        obj = lines[1].strip().split(' ')[1:]
        unused_P1 = np.array(obj, dtype=np.float32)

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)

        obj = lines[3].strip().split(' ')[1:]
        unused_P3 = np.array(obj, dtype=np.float32)

        obj = lines[4].strip().split(' ')[1:]
        unused_P4 = np.array(obj, dtype=np.float32)

        obj = lines[5].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32).reshape(3, 3)

        rect_4x4 = np.zeros([4, 4], dtype=R0.dtype)
        rect_4x4[3, 3] = 1
        rect_4x4[:3, :3] = R0

        obj = lines[6].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32).reshape(3, 4)
        Tr_velo_to_cam = Waymo._extend_matrix(Tr_velo_to_cam)

        world_cam = np.transpose(rect_4x4 @ Tr_velo_to_cam)
        cam_img = np.transpose(np.vstack((P2.reshape(3, 4), [0, 0, 0, 1])))

        return {'world_cam': world_cam, 'cam_img': cam_img}

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return WaymoSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        cfg = self.cfg
        dataset_path = cfg.dataset_path
        file_list = []

        if split in ['train', 'training']:
            return self.train_files
            seq_list = cfg.training_split
        elif split in ['test', 'testing']:
            return self.test_files
        elif split in ['val', 'validation']:
            return self.val_files
        elif split in ['all']:
            return self.train_files + self.val_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        raise NotImplementedError()

    def save_test_result(results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        raise NotImplementedError()


class WaymoSplit():

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
        label_path = ("label_all".join(pc_path.rsplit("velodyne", 1))).replace(
            '.bin', '.txt')
        calib_path = "calib".join(label_path.rsplit("label_all", 1))

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


class Object3d(BEVBox3D):

    def __init__(self, center, size, label, calib):
        # ground truth files doesn't have confidence value.
        confidence = float(label[15]) if label.__len__() == 16 else -1.0

        world_cam = calib['world_cam']
        cam_img = calib['cam_img']

        # kitti boxes are pointing backwards
        yaw = float(label[14]) - np.pi
        yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi

        self.truncation = float(label[1])
        self.occlusion = float(
            label[2]
        )  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown

        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(
            label[6]), float(label[7])),
                              dtype=np.float32)

        super().__init__(center, size, yaw, label[0], confidence, world_cam,
                         cam_img)

        self.yaw = float(label[14])

    def get_difficulty(self):
        """The method determines difficulty level of the object, such as Easy,
        Moderate, or Hard.
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

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.label_class, self.truncation, self.occlusion, self.alpha, self.box2d, self.size[2], self.size[0], self.size[1],
                        self.center, self.yaw)
        return print_str

    def to_kitti_format(self):
        """This method transforms the class to kitti format."""
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.label_class, self.truncation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.size[2], self.size[0], self.size[1], self.center[0], self.center[1], self.center[2],
                       self.yaw)
        return kitti_str


DATASET._register_module(Waymo)
