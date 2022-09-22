import numpy as np
import os
import logging
import open3d as o3d

from pathlib import Path
from os.path import join, exists
from glob import glob

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class KITTI360(BaseDataset):
    """This class is used to create a dataset based on the KITTI 360
    dataset, and used in visualizer, training, or testing.
    """

    def __init__(self,
                 dataset_path,
                 name='KITTI360',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 class_weights=[
                     3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                     650464, 791496, 88727, 1284130, 229758, 2272837
                 ],
                 num_points=40960,
                 ignored_label_inds=[],
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use (parent directory of data_3d_semantics).
            name: The name of the dataset (KITTI360 in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            class_weights: The class weights to use in the dataset.
            num_points: The maximum number of points to use when splitting the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         test_result_folder=test_result_folder,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         **kwargs)

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        if not os.path.exists(
                os.path.join(
                    dataset_path,
                    'data_3d_semantics/train/2013_05_28_drive_train.txt')):
            raise ValueError(
                "Invalid Path, make sure dataset_path is the parent directory of data_3d_semantics."
            )

        with open(
                os.path.join(
                    dataset_path,
                    'data_3d_semantics/train/2013_05_28_drive_train.txt'),
                'r') as f:
            train_paths = f.read().split('\n')[:-1]
            train_paths = [os.path.join(dataset_path, p) for p in train_paths]

        with open(
                os.path.join(
                    dataset_path,
                    'data_3d_semantics/train/2013_05_28_drive_val.txt'),
                'r') as f:
            val_paths = f.read().split('\n')[:-1]
            val_paths = [os.path.join(dataset_path, p) for p in val_paths]

        self.train_files = train_paths
        self.val_files = val_paths
        self.test_files = sorted(
            glob(
                os.path.join(dataset_path,
                             'data_3d_semantics/test/*/static/*.ply')))

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'ceiling',
            1: 'floor',
            2: 'wall',
            3: 'beam',
            4: 'column',
            5: 'window',
            6: 'door',
            7: 'table',
            8: 'chair',
            9: 'sofa',
            10: 'bookcase',
            11: 'board',
            12: 'clutter'
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return KITTI360Split(self, split=split)

    def get_split_list(self, split):
        if split in ['train', 'training']:
            return self.train_files
        elif split in ['val', 'validation']:
            return self.val_files
        elif split in ['test', 'testing']:
            return self.test_files
        elif split == 'all':
            return self.train_files + self.val_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self, attr):

        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
    """

    def save_test_result(self, results, attr):

        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(pred)

        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(path, self.name, name + '.npy')
        make_dir(Path(store_path).parent)
        np.save(store_path, pred)
        log.info("Saved {} in {}.".format(name, store_path))


class KITTI360Split(BaseDatasetSplit):
    """This class is used to create a split for KITTI360 dataset.

    Initialize the class.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]

        pc = o3d.t.io.read_point_cloud(pc_path)

        points = pc.point['positions'].numpy().astype(np.float32)
        feat = pc.point['colors'].numpy().astype(np.float32)
        labels = pc.point['semantic'].numpy().astype(np.int32).reshape((-1,))

        data = {
            'point': points,
            'feat': feat,
            'label': labels,
        }

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.pkl', '')

        pc_path = str(pc_path)
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': split}
        return attr


DATASET._register_module(KITTI360)
