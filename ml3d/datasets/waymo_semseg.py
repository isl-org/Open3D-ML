import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
import logging
import yaml

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import Config, make_dir, DATASET
from .utils import BEVBox3D

log = logging.getLogger(__name__)


class WaymoSemSeg(BaseDataset):
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
        self.num_classes = 23
        self.label_to_names = self.get_label_to_names()
        self.shuffle = kwargs.get('shuffle', False)

        self.all_files = sorted(
            glob(join(cfg.dataset_path, 'velodyne', '*.bin')))
        self.train_files = []
        self.val_files = []

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
        classes = "Undefined, Car, Truck, Bus, Other Vehicle, Motorcyclist, Bicyclist, Pedestrian, Sign, Traffic Light, Pole, Construction Cone, Bicycle, Motorcycle, Building, Vegetation, Tree Trunk, Curb, Road, Lane Marker, Other Ground, Walkable, Sidewalk"
        classes = classes.replace(', ', ',').split(',')
        label_to_names = {}
        for i in range(len(classes)):
            label_to_names[i] = classes[i]

        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            pc: pointcloud data with shape [N, 8], where
                the format is x,y,z,intensity,elongation,timestamp,instance, semantic_label.
        """
        return np.fromfile(path, dtype=np.float32).reshape(-1, 8)

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


class WaymoSplit(BaseDatasetSplit):

    def __init__(self, dataset, split='train'):
        super().__init__(dataset, split=split)
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

        pc = self.dataset.read_lidar(pc_path)
        feat = pc[:, 3:6]
        label = pc[:, 7].astype(np.int32)
        pc = pc[:, :3]

        data = {
            'point': pc,
            'feat': feat,
            'label': label,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr


DATASET._register_module(WaymoSemSeg)