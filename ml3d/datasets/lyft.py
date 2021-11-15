import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
import logging
import yaml
from scipy.spatial.transform import Rotation as R

from .base_dataset import BaseDataset
from ..utils import Config, make_dir, DATASET
from .utils import BEVBox3D

log = logging.getLogger(__name__)


class Lyft(BaseDataset):
    """This class is used to create a dataset based on the Lyft dataset, and
    used in object detection, visualizer, training, or testing.

    The Lyft level 5 dataset is best suited for self-driving applications.
    """

    def __init__(self,
                 dataset_path,
                 info_path=None,
                 name='Lyft',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            info_path: The path to the file that includes information about
            the dataset. This is default to dataset path if nothing is
            provided.
            name: The name of the dataset (Lyft in this case).
            cache_dir: The directory where the cache is stored.
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
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
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
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()

        return np.fromfile(path, dtype=np.float32).reshape(-1, 5)

    @staticmethod
    def read_label(info, calib):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        mask = info['num_lidar_pts'] != 0
        boxes = info['gt_boxes'][mask]
        names = info['gt_names'][mask]

        objects = []
        for name, box in zip(names, boxes):
            center = [float(box[0]), float(box[1]), float(box[2])]
            size = [float(box[3]), float(box[5]), float(box[4])]
            ry = float(box[6])

            yaw = ry - np.pi
            yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi

            world_cam = calib['world_cam']

            objects.append(BEVBox3D(center, size, yaw, name, -1.0, world_cam))
            objects[-1].yaw = ry

        return objects

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return LyftSplit(self, split=split)

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
        if split in ['train', 'training']:
            return self.train_info
        elif split in ['test', 'testing']:
            return self.test_info
        elif split in ['val', 'validation']:
            return self.val_info

        raise ValueError("Invalid split {}".format(split))

    def is_tested():
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the attribute is stored; else, returns false.
        """
        pass

    def save_test_result():
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
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

        world_cam = np.eye(4)
        world_cam[:3, :3] = R.from_quat(info['lidar2ego_rot']).as_matrix()
        world_cam[:3, -1] = info['lidar2ego_tr']
        calib = {'world_cam': world_cam.T}

        pc = self.dataset.read_lidar(lidar_path)

        data = {'point': pc, 'feat': None, 'calib': calib}

        if self.split not in ["test", "testing"]:
            data['bounding_boxes'] = self.dataset.read_label(info, calib)

        return data

    def get_attr(self, idx):
        info = self.infos[idx]
        pc_path = info['lidar_path']
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(Lyft)
