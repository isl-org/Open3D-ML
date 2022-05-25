import os
import pickle
from os.path import join
from pathlib import Path
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_dataset import BaseDataset
from ..utils import DATASET
from .utils import BEVBox3D
import open3d as o3d

log = logging.getLogger(__name__)


class NuScenesSemSeg(BaseDataset):
    """This class is used to create a dataset based on the NuScenes 3D dataset,
    and used in object detection, visualizer, training, or testing.

    The NuScenes 3D dataset is best suited for autonomous driving applications.
    """

    def __init__(self,
                 dataset_path,
                 info_path=None,
                 name='NuScenes',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            info_path: The path to the file that includes information about the
                dataset. This is default to dataset path if nothing is provided.
            name: The name of the dataset (NuScenes in this case).
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
        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)

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
            1: 'barrier',
            2: 'bicycle',
            3: 'bus',
            4: 'car',
            5: 'construction_vehicle',
            6: 'motorcycle',
            7: 'pedestrian',
            8: 'traffic_cone',
            9: 'trailer',
            10: 'truck',
            11: 'driveable_surface',
            12: 'other_flat',
            13: 'sidewalk',
            14: 'terrain',
            15: 'manmade',
            16: 'vegetation'
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
    def read_lidarseg(path):
        """Reads semantic data from the path provided.

        Returns:
            A data object with semantic information.
        """
        assert Path(path).exists()

        return np.fromfile(path, dtype=np.uint8).reshape(-1,).astype(np.int32)

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return NuScenesSemSegSplit(self, split=split)

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
            If the dataum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        pass

    def save_test_result():
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
                attribute passed.
            attr: The attributes that correspond to the outputs passed in
                results.
        """
        pass


class NuScenesSemSegSplit():

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg

        self.infos = dataset.get_split_list(split)
        self.path_list = []
        for info in self.infos:
            self.path_list.append(info['lidar_path'])

        log.info("Found {} pointclouds for {}".format(len(self.infos), split))

        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.infos)

    def get_data(self, idx):
        info = self.infos[idx]
        lidar_path = info['lidar_path']
        lidarseg_path = info['lidarseg_path']

        pc = self.dataset.read_lidar(lidar_path)
        feat = pc[:, 3:4]
        pc = pc[:, :3]
        lidarseg = self.dataset.read_lidarseg(lidarseg_path)

        data = {'point': pc, 'feat': feat, 'label': lidarseg}

        return data

    def get_attr(self, idx):
        info = self.infos[idx]
        pc_path = info['lidar_path']
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(NuScenesSemSeg)
