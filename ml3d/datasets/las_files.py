import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
import logging
import json
from .base_dataset import BaseDataset
from ..utils import make_dir, DATASET
from pyproj import Proj

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class woolpert_json(BaseDataset):
    """
    Toronto3D dataset, used in visualizer, training, or test
    """

    def __init__(self,
                 dataset_path,
                 name='woolpert_json',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[35391894., 1449308., 4650919., 18252779., 589856., 743579.,],
                 ignored_label_inds=[0],
                 train_files=['Block_000014.json','Block_000015.json','Block_000017.json','Block_000019.json','Block_000020.json','Block_000021.json','Block_000025.json'],
                 val_files=['Block_000012.json'],
                 test_files=['Block_000012.json'],
                 test_result_folder='./test',
                 **kwargs):
        """
        Initialize
        Args:
            dataset_path (str): path to the dataset
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
                         ignored_label_inds=ignored_label_inds,
                         train_files=train_files,
                         test_files=test_files,
                         val_files=val_files,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()
        self.names_to_labels = self.get_names_to_labels()
        self.dataset_path = cfg.dataset_path
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(cfg.ignored_label_inds)

        self.train_files = [
            join(self.cfg.dataset_path, f) for f in cfg.train_files
        ]
        self.val_files = [join(self.cfg.dataset_path, f) for f in cfg.val_files]
        self.test_files = [
            join(self.cfg.dataset_path, f) for f in cfg.test_files
        ]

    @staticmethod
    def get_label_to_names():
        # label_to_names = {
        #     0: 'Unclassified',
        #     1: 'Ground',
        #     2: 'Road_markings',
        #     3: 'Natural',
        #     4: 'Building',
        #     5: 'Utility_line',
        #     6: 'Pole',
        #     7: 'Car',
        #     8: 'Fence'
        # }
        label_to_names = {
             0: 'Poles (transmission and distribution)',
             1: 'Vegetations high (trees)',
             2: 'Road',
             3: 'Vegetation med (bushes)',
             4: 'Traffic sign',
             5: 'Conductors (wires)',
             6: 'Building',
             7: 'Vegetation low (grass)',
             8: 'Car'
        }
        return label_to_names
    def get_names_to_labels(self):
        name_to_lable = {}
        for key in self.label_to_names.keys():
            name_to_lable[self.label_to_names[key]]=key
        return name_to_lable

    def get_split(self, split):
        return woolpert_json_Split(self, split=split)

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


class woolpert_json_Split():

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

        self.UTM_OFFSET = [330321,4434969,880]

        self.cache_in_memory = self.cfg.get('cache_in_memory', False)
        if self.cache_in_memory:
            self.data_list = [None] * len(self.path_list)

    def __len__(self):
        return len(self.path_list)

    def read_json_v1(self,pc_path,create_lables_dict=False):
        myProj = Proj("+proj=utm +zone=17 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        names_to_labels = {}
        labels_to_names = {}
        with open(pc_path) as json_file:
            json_1 = json.load(json_file)
        is_first = True
        for instance in json_1['result']['data']:
            points_i = np.array([[myProj(p['lon'], p['lat'])[0],myProj(p['lon'], p['lat'])[1], p['h']] for p in instance['indexs']]).astype(np.float64)
            points_i = points_i - self.UTM_OFFSET
            points_i = np.float32(points_i)
            feat_i = np.zeros(points_i.shape, dtype=np.float32)
            category = instance['attr']['label'][0]
            if create_lables_dict:
                if category not in names_to_labels.keys():
                    c_label = len(names_to_labels.keys())
                    names_to_labels[category] = c_label
                else:
                    c_label = names_to_labels[category]
            else:
                c_label =  self.dataset.names_to_labels[category]
                print('name:{}, id:{}'.format(category,c_label))
            labels_i = np.array([c_label] * len(points_i), dtype=np.int32).T
            if is_first:
                points = points_i
                feat = feat_i
                labels = labels_i
                is_first = False
            else:
                points = np.vstack((points_i,points))
                feat = np.vstack((feat_i,feat))
                labels = np.hstack((labels_i,labels))
        data = {'point': points, 'feat': feat, 'label': labels}
        if create_lables_dict:
            label_to_names =  {value:key for (key,value) in names_to_labels.items()}
            self.dataset.label_to_names = label_to_names
        return data

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))
        if self.cache_in_memory:
            if self.data_list[idx] is not None:
                data = self.data_list[idx]
            else:
                #data = PlyData.read(pc_path)['vertex']
                data = self.read_json_v1(pc_path)
                self.data_list[idx] = data
        else:
            #data = PlyData.read(pc_path)['vertex']
            data = self.read_json_v1(pc_path)

        # points = np.vstack(
        #     (data['x'], data['y'], data['z'])).astype(np.float64).T
        # points = points - self.UTM_OFFSET
        # points = np.float32(points)
        #
        # feat = np.zeros(points.shape, dtype=np.float32)
        # feat[:, 0] = data['red']
        # feat[:, 1] = data['green']
        # feat[:, 2] = data['blue']
        #
        # labels = np.array(data['classification'], dtype=np.int32)
        # labels = np.clip(labels,0,8)
        # data = {'point': points, 'feat': feat, 'label': labels}
        #
        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(woolpert_json)
