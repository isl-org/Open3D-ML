import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
import logging

from sklearn.neighbors import KDTree
import yaml

from .base_dataset import BaseDataset
from .utils import DataProcessing
from ..utils import make_dir, DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class SemanticKITTI(BaseDataset):
    """
    SemanticKITTI dataset, used in visualizer, training, or test
    """

    def __init__(self,
                 dataset_path,
                 name='SemanticKITTI',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 class_weights=[
                     55437630, 320797, 541736, 2578735, 3274484, 552662, 184064,
                     78858, 240942562, 17294618, 170599734, 6369672, 230413074,
                     101130274, 476491114, 9833174, 129609852, 4506626, 1168181
                 ],
                 ignored_label_inds=[0],
                 test_result_folder='./test',
                 test_split=[
                     '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                     '21'
                 ],
                 training_split=[
                     '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
                 ],
                 validation_split=['08'],
                 all_split=[
                     '00', '01', '02', '03', '04', '05', '06', '07', '09', '08',
                     '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                     '20', '21'
                 ],
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

        data_config = join(dirname(abspath(__file__)), '_resources/',
                           'semantic-kitti.yaml')
        DATA = yaml.safe_load(open(data_config, 'r'))
        remap_dict = DATA["learning_map_inv"]

        # make lookup table for mapping
        max_key = max(remap_dict.keys())
        remap_lut = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

        remap_dict_val = DATA["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(
            remap_dict_val.values())

        self.remap_lut_val = remap_lut_val
        self.remap_lut = remap_lut

    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: 'unlabeled',
            1: 'car',
            2: 'bicycle',
            3: 'motorcycle',
            4: 'truck',
            5: 'other-vehicle',
            6: 'person',
            7: 'bicyclist',
            8: 'motorcyclist',
            9: 'road',
            10: 'parking',
            11: 'sidewalk',
            12: 'other-ground',
            13: 'building',
            14: 'fence',
            15: 'vegetation',
            16: 'trunk',
            17: 'terrain',
            18: 'pole',
            19: 'traffic-sign'
        }
        return label_to_names

    def get_split(self, split):
        return SemanticKITTISplit(self, split=split)

    def is_tested(self, attr):
        cfg = self.cfg
        name = attr['name']
        name_seq, name_points = name.split("_")
        test_path = join(cfg.test_result_folder, 'sequences')
        save_path = join(test_path, name_seq, 'predictions')
        test_file_name = name_points
        store_path = join(save_path, name_points + '.label')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        cfg = self.cfg
        name = attr['name']
        name_seq, name_points = name.split("_")

        test_path = join(cfg.test_result_folder, 'sequences')
        make_dir(test_path)
        save_path = join(test_path, name_seq, 'predictions')
        make_dir(save_path)
        test_file_name = name_points

        pred = results['predict_labels']

        store_path = join(save_path, name_points + '.label')
        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1
        pred = self.remap_lut[pred].astype(np.uint32)
        pred.tofile(store_path)

    def save_test_result_kpconv(self, results, inputs):
        cfg = self.cfg
        for j in range(1):
            name = inputs['attr']['name']
            name_seq, name_points = name.split("_")

            test_path = join(cfg.test_result_folder, 'sequences')
            make_dir(test_path)
            save_path = join(test_path, name_seq, 'predictions')
            make_dir(save_path)

            test_file_name = name_points

            proj_inds = inputs['data'].reproj_inds[0]
            probs = results[proj_inds, :]

            pred = np.argmax(probs, 1)

            store_path = join(save_path, name_points + '.label')
            pred = pred + 1
            pred = remap_lut[pred].astype(np.uint32)
            pred.tofile(store_path)

    def get_split_list(self, split):
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
            pc_path = join(dataset_path, 'dataset', 'sequences', seq_id,
                           'velodyne')
            file_list.append(
                [join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        file_list = np.concatenate(file_list, axis=0)

        return file_list


class SemanticKITTISplit():

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        self.remap_lut_val = dataset.remap_lut_val

        if split == 'test':
            dataset.test_list = path_list

        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        points = DataProcessing.load_pc_kitti(pc_path)

        dir, file = split(pc_path)
        label_path = join(dir, '../labels', file[:-4] + '.label')
        if not exists(label_path):
            labels = np.zeros(np.shape(points)[0], dtype=np.int32)
            if self.split not in ['test', 'all']:
                raise FileNotFoundError(f' Label file {label_path} not found')

        else:
            labels = DataProcessing.load_label_kitti(
                label_path, self.remap_lut_val).astype(np.int32)

        data = {
            'point': points[:, 0:3],
            'feat': None,
            'label': labels,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        dir, file = split(pc_path)
        _, seq = split(split(dir)[0])
        name = '{}_{}'.format(seq, file[:-4])

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr


DATASET._register_module(SemanticKITTI)
