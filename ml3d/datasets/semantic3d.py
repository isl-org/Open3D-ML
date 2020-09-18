import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from tqdm import tqdm
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging

from .utils import DataProcessing as DP
from .base_dataset import BaseDataset
from ..utils import make_dir, DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class Semantic3D(BaseDataset):
    """
    SemanticKITTI dataset, used in visualizer, training, or test
    """

    def __init__(self,
                 dataset_path,
                 name='Semantic3D',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 prepro_grid_size=0.06,
                 class_weights=[
                     5181602, 5012952, 6830086, 1311528, 10476365, 946982,
                     334860, 269353
                 ],
                 ignored_label_inds=[0],
                 val_split=1,
                 test_result_folder='./test',
                 pc_size_limit=500, # In mega bytes.
                 big_pc_path='./logs/Semantic3D/',
                 **kwargs):
        """
        Initialize
        Args:
            dataset_path: path to the dataset
            kwargs:
        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         num_points=num_points,
                         prepro_grid_size=prepro_grid_size,
                         ignored_label_inds=ignored_label_inds,
                         val_split=val_split,
                         test_result_folder=test_result_folder,
                         pc_size_limit=pc_size_limit,
                         big_pc_path=big_pc_path,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = {
            0: 'unlabeled',
            1: 'man-made terrain',
            2: 'natural terrain',
            3: 'high vegetation',
            4: 'low vegetation',
            5: 'buildings',
            6: 'hard scape',
            7: 'scanning artefacts',
            8: 'cars'
        }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        self.all_files = glob.glob(str(Path(self.cfg.dataset_path) / '*.txt'))

        self.train_files = [
            f for f in self.all_files if exists(
                str(Path(f).parent / Path(f).name.replace('.txt', '.labels')))
        ]
        self.test_files = [
            f for f in self.all_files if f not in self.train_files
        ]

        self.all_split = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.val_split = cfg.val_split

        self.train_files = np.sort(self.train_files)
        self.test_files = np.sort(self.test_files)
        self.val_files = []

        for i, file_path in enumerate(self.train_files):
            if self.all_split[i] == self.val_split:
                self.val_files.append(file_path)

        self.train_files = np.sort(
            [f for f in self.train_files if f not in self.val_files])

        train_big_files = {}
        train_files_parts = []
        for f in self.train_files:
            size = Path(f).stat().st_size / 1e6
            if size <= cfg.pc_size_limit:
                train_files_parts.append(f)
                continue
            parts = int(size/cfg.pc_size_limit) + 1
            train_big_files[f] = parts
            name = Path(f).name
            for i in range(parts):
                train_files_parts.append(cfg.big_pc_path + name.replace('.txt', '_part_{}.txt'.format(i)))

        self.train_files = train_files_parts
        self.train_big_files = train_big_files

        val_files_parts = []
        val_big_files = {}
        for f in self.val_files:
            size = Path(f).stat().st_size / 1e6
            if size <= cfg.pc_size_limit:
                val_files_parts.append(f)
                continue
            parts = int(size/cfg.pc_size_limit) + 1
            val_big_files[f] = parts
            name = Path(f).name
            for i in range(parts):
                val_files_parts.append(cfg.big_pc_path + name.replace('.txt', '_part_{}.txt'.format(i)))

        self.val_files = val_files_parts
        self.val_big_files = val_big_files

    def get_split(self, split):
        return Semantic3DSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError("Invalid split {}".format(split))

        return files

    def get_big_pc_list(self, split):
        if split in ['train', 'training']:
            files = self.train_big_files
        elif split in ['val', 'validation']:
            files = self.val_big_files
        else:
            files = []

        return files

    def save_test_result(self, results, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(self.label_to_names[pred])

        store_path = join(path, name + '.npy')
        np.save(store_path, pred)


class Semantic3DSplit():

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

        big_pc_list = dataset.get_big_pc_list(split)
        if len(big_pc_list):
            make_dir(self.cfg.big_pc_path)
            self.split_big_pc(big_pc_list)

    def __len__(self):
        return len(self.path_list)

    def split_big_pc(self, big_pc_list):
        cfg = self.cfg
        log.info("Splitting large point clouds.")
        for key, parts in tqdm(big_pc_list.items()):
            flag_exists = 1
            for i in range(parts):
                name = join(cfg.big_pc_path, Path(key).name.replace('.txt', '_part_{}.txt'.format(i)))
                if not exists(name):
                    flag_exists = 0
                    break
            if(flag_exists):
                continue

            log.info("Splitting {} into {} parts".format(Path(key).name, parts))
            pc = pd.read_csv(key,
                            header=None,
                            delim_whitespace=True,
                            dtype=np.float32).values

            labels = pd.read_csv(key.replace(".txt", ".labels"),
                                header=None,
                                delim_whitespace=True,
                                dtype=np.int32).values
            labels = np.array(labels, dtype=np.int32).reshape((-1,))

            points = pc[:, 0:3]
            axis_range = []
            for i in range(2):
                min_i = np.min(points[:, i])
                max_i = np.max(points[:, i])
                axis_range.append(max_i - min_i)
            axis = np.argmax(axis_range)

            inds = pc[:, axis].argsort()
            pc = pc[inds]
            labels = labels[inds]
            pcs = np.array_split(pc, parts)
            lbls = np.array_split(labels, parts)
            for i in range(parts):
                name = join(cfg.big_pc_path, Path(key).name.replace('.txt', '_part_{}.txt'.format(i)))
                name_lbl = name.replace('.txt', '.labels')

                shuf = np.arange(pcs[i].shape[0])
                np.random.shuffle(shuf)

                np.savetxt(name, pcs[i][shuf])
                np.savetxt(name_lbl, lbls[i][shuf])

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        pc = pd.read_csv(pc_path,
                         header=None,
                         delim_whitespace=True,
                         dtype=np.float32).values

        points = pc[:, 0:3]
        feat = pc[:, [4, 5, 6]]
        intensity = pc[:, 3]

        points = np.array(points, dtype=np.float32)
        feat = np.array(feat, dtype=np.float32)
        intensity = np.array(intensity, dtype=np.float32)

        if (self.split != 'test'):
            labels = pd.read_csv(pc_path.replace(".txt", ".labels"),
                                 header=None,
                                 delim_whitespace=True,
                                 dtype=np.int32).values
            labels = np.array(labels, dtype=np.int32).reshape((-1,))
        else:
            labels = np.zeros((points.shape[0],), dtype=np.int32)

        data = {'point': points, 'feat': feat, 'intensity': intensity, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(Semantic3D)
