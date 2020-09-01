import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
import torch
import logging

from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler
from sklearn.neighbors import KDTree
import yaml

from .utils import DataProcessing
from ..utils import make_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class SemanticKITTISplit(Dataset):
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
            labels = np.zeros(np.shape(points)[0], dtype=np.uint8)
            if self.split not in ['test', 'all']:
                raise ValueError("label file not found for {}".format(label_path))
        else:
            labels = DataProcessing.load_label_kitti(label_path, self.remap_lut_val)

        data = {'point': points, 
                'feat' : None,
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


class SemanticKITTI:
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = 'SemanticKITTI'
        self.label_to_names = {
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
        self.num_classes = len(self.label_to_names)
        data_config = join(cfg.dataset_path, 'semantic-kitti.yaml')
        DATA = yaml.safe_load(open(data_config, 'r'))
        remap_dict = DATA["learning_map_inv"]

        # make lookup table for mapping
        max_key = max(remap_dict.keys())
        remap_lut = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

        remap_dict_val = DATA["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val


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
        pred = pred + 1
        pred = remap_lut[pred].astype(np.uint32)
        # pred.tofile(store_path)

    def save_test_result_kpconv(self, results, inputs):
        cfg = self.cfg
        for j in range(1):
            # name = inputs['attr']['name']
            name = inputs['attr']['name']
            # print(name)
            name_seq, name_points = name.split("_")

            test_path = join(cfg.test_result_folder, 'sequences')
            make_dir(test_path)
            save_path = join(test_path, name_seq, 'predictions')
            make_dir(save_path)

            test_file_name = name_points
            # proj_inds = inputs['data']['proj_inds'][j].cpu().numpy()
            
            proj_inds = inputs['data'].reproj_inds[0]
            # proj_inds = inputs.proj_inds
            probs = results[proj_inds, :]
            # probs = results[j][proj_inds, :]
          
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
            pc_path = join(dataset_path, seq_id, 'velodyne')
            file_list.append(
                [join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        file_list = np.concatenate(file_list, axis=0)

        return file_list
