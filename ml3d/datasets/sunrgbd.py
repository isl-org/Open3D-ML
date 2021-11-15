import numpy as np
import os, pickle
from os.path import join
from pathlib import Path
import logging

from .base_dataset import BaseDataset
from ..utils import DATASET
from .utils import BEVBox3D

log = logging.getLogger(__name__)


class SunRGBD(BaseDataset):
    """SunRGBD 3D dataset for Object Detection, used in visualizer, training, or
    test.
    """

    def __init__(self,
                 dataset_path,
                 name='SunRGBD',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 **kwargs):
        """Initialize the dataset by passing the dataset and other details.

        Args:
            dataset_path (str): The path to the dataset to use.
            name (str): The name of the dataset (SunRGBD in this case).
            cache_dir (str): The directory where the cache is stored.
            use_cache (bool): Indicates if the dataset should be cached.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 18

        self.classes = [
            'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
            'night_stand', 'bookshelf', 'bathtub'
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}

        self.label_to_names = self.get_label_to_names()

        available_idx = []
        files = os.listdir(join(dataset_path, 'depth'))
        for f in files:
            if f.endswith('.npy'):
                available_idx.append(f.split('.')[0])

        train_files = open(join(self.dataset_path,
                                'train_data_idx.txt')).read().split('\n')
        val_files = open(join(self.dataset_path,
                              'val_data_idx.txt')).read().split('\n')

        self.train_idx = []
        self.val_idx = []
        for idx in available_idx:
            if idx in train_files:
                self.train_idx.append(idx)
            elif idx in val_files:
                self.val_idx.append(idx)

    def get_label_to_names(self):
        return self.label2cat

    @staticmethod
    def read_lidar(path):
        assert Path(path).exists()
        data = np.load(path).astype(np.float32)

        return data

    def read_label(self, path):
        assert Path(path).exists()

        bboxes = pickle.load(open(path, 'rb'))

        objects = []
        for box in bboxes:
            name = box[0]
            center = box[1:4]
            size = [box[4] * 2, box[6] * 2, box[5] * 2]  # w, h, l
            orientation = [box[7], box[8]]
            yaw = -1 * np.arctan(orientation[1] / orientation[0])

            if len(box) > 9:
                box2d = [box[9], box[10], box[9] + box[11], box[10] + box[12]]
            else:
                box2d = []

            objects.append(Object3d(name, center, size, yaw, box2d))

        return objects

    def get_split(self, split):
        return SunRGBDSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['train', 'training']:
            return self.train_idx
        elif split in ['test', 'testing']:
            return self.val_idx
        elif split in ['val', 'validation']:
            return self.val_idx

        raise ValueError("Invalid split {}".format(split))

    def is_tested(self):
        pass

    def save_test_result(self):
        pass


class SunRGBDSplit():

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg

        self.path_list = dataset.get_split_list(split)

        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        idx = self.path_list[idx]

        pc = self.dataset.read_lidar(
            join(self.cfg.dataset_path, f'depth/{idx}.npy'))
        feat = pc[:, 3:]
        pc = pc[:, :3]

        bboxes = self.dataset.read_label(
            join(self.cfg.dataset_path, f'label/{idx}.pkl'))

        data = {
            'point': pc,
            'feat': feat[:, [2, 1, 0]],
            'calib': None,
            'bounding_boxes': bboxes,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


class Object3d(BEVBox3D):
    """Stores object specific details like bbox coordinates."""

    def __init__(self, name, center, size, yaw, box2d):
        super().__init__(center, size, yaw, name, -1.0)

        self.occlusion = 0.0
        self.box2d = box2d


DATASET._register_module(SunRGBD)
