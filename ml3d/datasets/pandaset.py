import os
from os.path import join
from pathlib import Path
import logging
import numpy as np
import pandas as pd

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class Pandaset(BaseDataset):
    """This class is used to create a dataset based on the Pandaset autonomous
    driving dataset.

    PandaSet aims to promote and advance research and development in autonomous
    driving and machine learning. The first open-source AV dataset available for
    both academic and commercial use, PandaSet combines Hesai’s best-in-class
    LiDAR sensors with Scale AI’s high-quality data annotation.

    PandaSet features data collected using a forward-facing LiDAR with
    image-like resolution (PandarGT) as well as a mechanical spinning LiDAR
    (Pandar64). The collected data was annotated with a combination of cuboid
    and segmentation annotation (Scale 3D Sensor Fusion Segmentation).

    It features::

      - 48,000+ camera images
      - 16,000+ LiDAR sweeps
      - 100+ scenes of 8s each
      - 28 annotation classes
      - 37 semantic segmentation labels
      - Full sensor suite: 1x mechanical spinning LiDAR, 1x forward-facing LiDAR, 6x cameras, On-board GPS/IMU

    Website: https://pandaset.org/
    Code: https://github.com/scaleapi/pandaset-devkit
    Download: https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset/data
    Data License: CC0: Public Domain (https://scale.com/legal/pandaset-terms-of-use)
    Citation: https://arxiv.org/abs/2112.12610
    """

    def __init__(self,
                 dataset_path,
                 name="Pandaset",
                 cache_dir="./logs/cache",
                 use_cache=False,
                 ignored_label_inds=[],
                 test_result_folder='./logs/test_log',
                 test_split=[
                     '115', '116', '117', '119', '120', '124', '139', '149',
                     '158'
                 ],
                 training_split=[
                     '001', '002', '003', '005', '011', '013', '015', '016',
                     '017', '019', '021', '023', '024', '027', '028', '029',
                     '030', '032', '033', '034', '035', '037', '038', '039',
                     '040', '041', '042', '043', '044', '046', '052', '053',
                     '054', '056', '057', '058', '064', '065', '066', '067',
                     '070', '071', '072', '073', '077', '078', '080', '084',
                     '088', '089', '090', '094', '095', '097', '098', '101',
                     '102', '103', '105', '106', '109', '110', '112', '113'
                 ],
                 validation_split=['122', '123'],
                 all_split=[
                     '001', '002', '003', '005', '011', '013', '015', '016',
                     '017', '019', '021', '023', '024', '027', '028', '029',
                     '030', '032', '033', '034', '035', '037', '038', '039',
                     '040', '041', '042', '043', '044', '046', '052', '053',
                     '054', '056', '057', '058', '064', '065', '066', '067',
                     '069', '070', '071', '072', '073', '077', '078', '080',
                     '084', '088', '089', '090', '094', '095', '097', '098',
                     '101', '102', '103', '105', '106', '109', '110', '112',
                     '113', '115', '116', '117', '119', '120', '122', '123',
                     '124', '139', '149', '158'
                 ],
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset.
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            ignored_label_inds: A list of labels that should be ignored in the dataset.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         test_split=test_split,
                         training_split=training_split,
                         validation_split=validation_split,
                         all_split=all_split,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            1: "Reflection",
            2: "Vegetation",
            3: "Ground",
            4: "Road",
            5: "Lane Line Marking",
            6: "Stop Line Marking",
            7: "Other Road Marking",
            8: "Sidewalk",
            9: "Driveway",
            10: "Car",
            11: "Pickup Truck",
            12: "Medium-sized Truck",
            13: "Semi-truck",
            14: "Towed Object",
            15: "Motorcycle",
            16: "Other Vehicle - Construction Vehicle",
            17: "Other Vehicle - Uncommon",
            18: "Other Vehicle - Pedicab",
            19: "Emergency Vehicle",
            20: "Bus",
            21: "Personal Mobility Device",
            22: "Motorized Scooter",
            23: "Bicycle",
            24: "Train",
            25: "Trolley",
            26: "Tram / Subway",
            27: "Pedestrian",
            28: "Pedestrian with Object",
            29: "Animals - Bird",
            30: "Animals - Other",
            31: "Pylons",
            32: "Road Barriers",
            33: "Signs",
            34: "Cones",
            35: "Construction Signs",
            36: "Temporary Construction Barriers",
            37: "Rolling Containers",
            38: "Building",
            39: "Other Static Object"
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return PandasetSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or 'all'.
        """
        cfg = self.cfg
        dataset_path = cfg.dataset_path
        file_list = []

        if split in ['train', 'training']:
            seq_list = cfg.training_split
        elif split in ['test', 'testing']:
            seq_list = cfg.test_split
        elif split in ['val', 'validation']:
            seq_list = cfg.validation_split
        elif split in ['all']:
            seq_list = cfg.all_split
        else:
            raise ValueError("Invalid split {}".format(split))

        for seq_id in seq_list:
            pc_path = join(dataset_path, seq_id, 'lidar')
            for f in np.sort(os.listdir(pc_path)):
                if f.split('.')[-1] == 'gz':
                    file_list.append(join(pc_path, f))

        return file_list

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the
            attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
                attribute passed.
            attr: The attributes that correspond to the outputs passed in
                results.
        """
        cfg = self.cfg
        pred = results['predict_labels']
        name = attr['name']

        test_path = join(cfg.test_result_folder, 'sequences')
        make_dir(test_path)
        save_path = join(test_path, name, 'predictions')
        make_dir(save_path)
        pred = results['predict_labels']

        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(save_path, name + '.label')

        pred = pred.astype(np.uint32)
        pred.tofile(store_path)


class PandasetSplit(BaseDatasetSplit):
    """This class is used to create a split for Pandaset dataset.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='train'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        label_path = pc_path.replace('lidar', 'annotations/semseg')

        points = pd.read_pickle(pc_path)
        labels = pd.read_pickle(label_path)

        intensity = points['i'].to_numpy().astype(np.float32)
        points = points.drop(columns=['i', 't', 'd']).to_numpy().astype(
            np.float32)
        labels = labels.to_numpy().astype(np.int32)

        data = {'point': points, 'intensity': intensity, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        value = (pc_path).split('/')[9]
        name = Path(pc_path).name.split('.')[0]
        name = value + '_' + name

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr


DATASET._register_module(Pandaset)
