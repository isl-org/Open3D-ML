import numpy as np
from os.path import join
from pathlib import Path
from glob import glob
import joblib
import logging

from .base_dataset import BaseDataset
from ..utils import DATASET
from .utils import BEVBox3D

log = logging.getLogger(__name__)


class MatterportObjects(BaseDataset):
    """This class is used to create a dataset based on the Matterport-Chair 
    dataset and other related datasets.

    The Matterport-Chair dataset is introduced in Sparse PointPillars as a
    chair detection task for an embodied agent in various homes in Matterport3D 
    (https://niessner.github.io/Matterport/). The training and test splits for
    Matterport-Chair are available on the Sparse PointPillars project webpage 
    (https://vedder.io/sparse_point_pillars) and code to generate Matterport-Chair
    can be used to generate datasets of other objects in Matterport3D 
    (https://github.com/kylevedder/MatterportDataSampling).

    Point clouds and bounding boxes are stored as numpy arrays serialized with 
    joblib. All coordinates are in the standard robot coordinate frame 
    (https://en.wikipedia.org/wiki/Right-hand_rule#Coordinates), with X forward,
    Y to the left, and Z up. All bounding boxes are assumed to only have a rotation 
    along the Z axis in the form of yaw (positive yaw is counterclockwise).

    Like with KITTI, before you use Matterport-Chair you should run
    scripts/collect_bboxes.py to generate the bbox dictionary for data augmentation,
    but with '--dataset_type MatterportObjects' specified.

    If you use this in your research, we ask that you please cite Sparse PointPillars 
    (https://github.com/kylevedder/SparsePointPillars#citation).
    """

    def __init__(self,
                 dataset_path,
                 name='MatterportObjects',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 val_split=5000,
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (MatterportObjects in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            val_split: The split value to get a set of images for training,
            validation, for testing.
            test_result_folder: Path to store test output.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         val_split=val_split,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 1
        self.label_to_names = self.get_label_to_names()

        self.all_files = glob(join(cfg.dataset_path, 'training', 'pc', '*.bin'))
        self.all_files.sort()
        # Ensures that the training and validation regions
        # of the dataset are not uniform distinct regions of index,
        # while still being deterministic
        self.rng.shuffle(self.all_files)
        self.train_files = []
        self.val_files = []

        for f in self.all_files:
            idx = int(Path(f).name.replace('.bin', ''))
            if idx < cfg.val_split:
                self.train_files.append(f)
            else:
                self.val_files.append(f)

        self.test_files = glob(join(cfg.dataset_path, 'test', 'pc', '*.bin'))
        self.test_files.sort()

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictonary object.

        Returns:
            A dict where keys are label numbers and values are the corresponding
            names.

            Names are extracted from Matterport3D's `metadata/category_mapping.tsv`'s
            "ShapeNetCore55" column. 
        """
        label_to_names = {
            0: 'chair',
            1: 'pillow',
            2: 'lamp',
            3: 'cabinet',
            4: 'table',
            5: 'sofa',
            6: 'bed',
            7: 'jar',
            8: 'tv or monitor',
            9: 'bench',
            10: 'trash_bin',
            11: 'tub',
            12: 'faucet',
            13: 'bottle',
            14: 'bookshelf',
            15: 'basket',
            16: 'clock',
            17: 'stove',
            18: 'washing_machine',
            19: 'microwave',
            20: 'bowl',
            21: 'flower pot',
            22: 'speaker',
            23: 'printer',
            24: 'telephone',
            25: 'computer keyboard',
            26: 'dishwasher',
            27: 'cup or mug',
            28: 'piano',
            29: 'suitcase',
            30: 'laptop',
            31: 'guitar',
            32: 'car',
            33: 'skateboard',
            34: 'camera',
            35: 'bicycle',
            36: 'watercraft',
            37: 'can',
            38: 'knife',
            39: 'wine bottle',
            40: 'tower',
            41: 'motorcycle',
            42: 'DontCare'
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()
        return joblib.load(path)

    @staticmethod
    def read_label(path):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        assert Path(path).exists()
        boxes = joblib.load(path)
        objects = []
        for b in boxes:
            name, img_left, img_top, img_right, img_bottom, center_x, center_y, center_z, l, w, h, yaw = b
            yaw = -np.deg2rad(np.float32(yaw))
            # image_bb = np.array([img_left, img_top, img_right, img_bottom])
            size = np.array([l, h, w],
                            np.float32)  # Weird order is what the BEV box takes
            center = np.array([center_x, center_y, center_z],
                              np.float32)  # Actual center of the box
            objects.append(BEVBox3D(center, size, yaw, name, 1))
        return objects

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return MatterportObjectsSplit(self, split=split)

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
            return self.train_files
        elif split in ['test', 'testing']:
            return self.test_files
        elif split in ['val', 'validation']:
            return self.val_files
        elif split in ['all']:
            return self.train_files + self.val_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then resturn the path where the
            attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, attrs):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
            attribute passed.
            attrs: The attributes that correspond to the outputs passed in
            results.
        """
        pass


class MatterportObjectsSplit():

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
        label_path = ("boxes".join(pc_path.rsplit("pc",
                                                  1))).replace('.bin', '.txt')

        pc = self.dataset.read_lidar(pc_path)
        label = self.dataset.read_label(label_path)

        data = {
            'point': pc,
            'calib': {},
            'bounding_boxes': label,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr


DATASET._register_module(MatterportObjects)
