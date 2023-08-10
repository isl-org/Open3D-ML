from .base_dataset import BaseDataset, BaseDatasetSplit
import glob
from pathlib import Path
import logging
from os.path import join, exists
from ..utils import make_dir, DATASET
import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)


class TUMFacade(BaseDataset):
    def __init__(self,
                 dataset_path,
                 info_path=None,
                 name='TUM_Facade',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 use_global=False,
                 **kwargs):

        """Initialize the function by passing the dataset and other details.

             Args:
                 dataset_path: The path to the dataset to use.
                 info_path: The path to the file that includes information about
                 the dataset. This is default to dataset path if nothing is
                 provided.
                 name: The name of the dataset (TUM_Facade in this case).
                 cache_dir: The directory where the cache is stored.
                 use_cache: Indicates if the dataset should be cached.
                 use_global: Inidcates if the dataset should be used in a loca or the global CRS

             Returns:
                 class: The corresponding class.
             """
        super().__init__(dataset_path=dataset_path,
                         info_path=info_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         use_global=use_global, # Diese habe ich selbst hinzugefügt
                         **kwargs)
        cfg = self.cfg
        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.label_to_names = self.get_label_to_names()
        self.use_global = cfg.use_global
        if info_path is None:
            self.info_path = dataset_path

        if self.use_global:
            # Finding all the training files
            self.trainFiles = glob.glob(str(Path(cfg.dataset_path) / 'pointclouds' / 'annotatedGlobalCRS' / 'training_files' / '*.pcd'))
            # Finding all the validation Files
            self.valFiles = glob.glob(str(Path(cfg.dataset_path) / 'pointclouds' / 'annotatedGlobalCRS' / 'validation_files' / '*.pcd'))
            # Finding all the test files
            self.testFiles = glob.glob(str(Path(cfg.dataset_path) / 'pointclouds' / 'annotatedGlobalCRS' / 'test_files' / '*.pcd'))

        elif not self.use_global:
            # Finding all the training files
            self.trainFiles = glob.glob(str(Path(cfg.dataset_path) / 'pointclouds' / 'annotatedLocalCRS' / 'training_files' / '*.pcd'))
            # Finding all the validation Files
            self.valFiles = glob.glob(str(Path(cfg.dataset_path) / 'pointclouds' / 'annotatedLocalCRS' / 'validation_files' / '*.pcd'))
            # Finding all the test files
            self.testFiles = glob.glob(str(Path(cfg.dataset_path) / 'pointclouds' / 'annotatedLocalCRS' / 'test_files' / '*.pcd'))

        else:
            raise ValueError("Invalid specification! use_global must either be True or False!")


    @staticmethod
    def get_label_to_names():  #
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and values are the corresponding
            names.
        """
        label_to_names = {
            0: 'not_assigned',
            1: 'wall',
            2: 'window',
            3: 'door',
            4: 'balcony',
            5: 'molding',
            6: 'deco',
            7: 'column',
            8: 'arch',
            9: 'drainpipe',
            10: 'stairs',
            11: 'ground_surface',
            12: 'terrain',
            13: 'roof',
            14: 'blinds',
            15: 'outer_ceiling_surface',
            16: 'interior',
            17: 'other'
        }
        return label_to_names

    def get_split(self, split):
        return TUMFacadeSplit(self, split=split)

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
            return self.trainFiles
        elif split in ['test', 'testing']:
            return self.testFiles
        elif split in ['val', 'validation']:
            return self.valFiles
        elif split in ['all']:
            return self.trainFiles + self.valFiles + self.testFiles
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self, attr):

       pass


    def save_test_result(self, results, attr):

        pass


class TUMFacadeSplit(BaseDatasetSplit):
    def __init__(self, dataset, split='train'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))
    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        data = o3d.t.io.read_point_cloud(pc_path).point
        points = data["positions"].numpy()
        points = np.float32(points)
        labels = data['classification'].numpy().astype(np.int32).reshape((-1,))
        data = {'point': points, 'feat': None, 'label': labels}
        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        pc_path = str(pc_path)
        name = pc_path.replace('.txt', '')
        parts = name.split("/")
        name = parts[-1]
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': split}
        return attr

DATASET._register_module(TUMFacade)