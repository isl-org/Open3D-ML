import open3d as o3d
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


class Argoverse(BaseDataset):
    """This class is used to create a dataset based on the Agroverse dataset,
    and used in object detection, visualizer, training, or testing.
    """

    def __init__(self,
                 dataset_path,
                 info_path=None,
                 name='Argoverse',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            info_path: The path to the file that includes information about the
                dataset. This is default to dataset path if nothing is provided.
            name: The name of the dataset.
            cache_dir: The directory where the cache will be stored.
            use_cache: Indicates if the dataset should be cached.

        Returns:
            class: The corresponding class.
        """
        if info_path is None:
            info_path = dataset_path

        super().__init__(dataset_path=dataset_path,
                         info_path=info_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 15
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

        if os.path.exists(join(info_path, 'infos_sample.pkl')):
            self.sample_info = pickle.load(
                open(join(info_path, 'infos_sample.pkl'), 'rb'))

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and values are the corresponding
            names.
        """
        label_to_names = {
            0: 'ignore',
            1: 'VEHICLE',
            2: 'PEDESTRIAN',
            3: 'ON_ROAD_OBSTACLE',
            4: 'LARGE_VEHICLE',
            5: 'BICYCLE',
            6: 'BICYCLIST',
            7: 'BUS',
            8: 'OTHER_MOVER',
            9: 'TRAILER',
            10: 'MOTORCYCLIST',
            11: 'MOPED',
            12: 'MOTORCYCLE',
            13: 'STROLLER',
            14: 'EMERGENCY_VEHICLE',
            15: 'ANIMAL'
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()

        data = np.asarray(o3d.io.read_point_cloud(path).points).astype(
            np.float32)

        return data

    @staticmethod
    def read_label(bboxes):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        objects = []
        for box in bboxes:
            name = box['label_class']
            center = box['center']
            size = [box['w'], box['h'], box['l']]

            box2d = box['2d_coord']

            yaw = np.pi / 2 + np.arctan(
                (box2d[0][0] - box2d[1][0]) / (box2d[0][1] - box2d[1][1]))
            objects.append(Object3d(center, size, yaw, name, box))

        return objects

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return ArgoverseSplit(self, split=split)

    def get_split_list(self, split):
        """Returns a dataset split.

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
        if split in ['train', 'training']:
            return self.train_info
        elif split in ['test', 'testing']:
            return self.test_info
        elif split in ['val', 'validation']:
            return self.val_info
        elif split in ['sample']:
            return self.sample_info

        raise ValueError("Invalid split {}".format(split))

    def is_tested(self):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the
            attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
            attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        pass


class ArgoverseSplit():
    """This class is used to create a split for Agroverse dataset.

    Initialize the class.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg

        infos = dataset.get_split_list(split)

        self.num_pc = 0
        self.path_list = []
        self.bboxes = []

        for info in infos:
            self.num_pc += info['num_pc']
            self.path_list += info['lidar_path']
            self.bboxes += info['bbox']

        log.info("Found {} pointclouds for {}".format(self.num_pc, split))

        self.split = split
        self.dataset = dataset

    def __len__(self):
        return self.num_pc

    def get_data(self, idx):
        lidar_path = self.path_list[idx]
        bboxes = self.bboxes[idx]

        pc = self.dataset.read_lidar(lidar_path)

        label = self.dataset.read_label(bboxes)

        data = {
            'point': pc,
            'feat': None,
            'calib': None,
            'bounding_boxes': label,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


class Object3d(BEVBox3D):
    """The class stores details that are object-specific, such as bounding box
    coordinates.
    """

    def __init__(self, center, size, yaw, name, box):
        super().__init__(center, size, yaw, name, -1.0)

        self.occlusion = box['occlusion']
        self.quaternion = box['quaternion']
        self.coords_3d = box['3d_coord']
        self.coords_2d = box['2d_coord']

    def generate_corners3d(self):
        """This generates a Corners 3D representation for the object, and
        returns the corners in 3D, such as (8, 3) corners of a Box3D in camera
        coordinates.
        """
        return self.coords_3d


DATASET._register_module(Argoverse)
