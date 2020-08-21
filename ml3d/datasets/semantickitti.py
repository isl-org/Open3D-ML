import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
import torch

from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler
from sklearn.neighbors import KDTree
import yaml

from .utils import DataProcessing
from ..utils import make_dir

BASE_DIR = dirname(abspath(__file__))

data_config = join(BASE_DIR, 'utils', 'semantic-kitti.yaml')
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


class SemanticKITTISplit(Dataset):
    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)

        if split == 'test':
            dataset.test_list = path_list

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        points = DataProcessing.load_pc_kitti(pc_path)
        if self.split != 'test':
            dir, file = split(pc_path)
            label_path = join(dir, '../labels', file[:-4] + '.label')
            labels = DataProcessing.load_label_kitti(label_path, remap_lut_val)
        else:
            labels = np.zeros(np.shape(points)[0], dtype=np.uint8)

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

        self.possibility = []
        self.min_possibility = []

    def get_split(self, split):
        return SemanticKITTISplit(self, split=split)

    def save_test_result(self, test_probs, test_scan_name):
        cfg = self.cfg

        test_path = join(cfg.test_result_folder, 'sequences')
        make_dir(test_path)
        save_path = join(test_path, test_scan_name, 'predictions')
        make_dir(save_path)

        for j in range(len(test_probs)):
            test_file_name = self.test_list[j]
            frame = test_file_name.split('/')[-1][:-4]
            proj_path = join(cfg.dataset_path, test_scan_name, 'proj')
            proj_file = join(proj_path, str(frame) + '_proj.pkl')
            if isfile(proj_file):
                with open(proj_file, 'rb') as f:
                    proj_inds = pickle.load(f)
            probs = test_probs[j][proj_inds[0], :]
            pred = np.argmax(probs, 1)

            store_path = join(test_path, test_scan_name, 'predictions',
                              str(frame) + '.label')
            pred = pred + 1
            pred = remap_lut[pred].astype(np.uint32)
            pred.tofile(store_path)

    def get_split_list(self, split):
        cfg = self.cfg
        dataset_path = cfg.dataset_path
        file_list = []

        if split == 'training':
            seq_list = cfg.training_split
        elif split == 'test':
            # seq_list = [str(cfg.test_split_number)]
            seq_list = cfg.test_split
        elif split == 'validation':
            seq_list = cfg.validation_split

        # self.prepro_randlanet(seq_list, split)

        for seq_id in seq_list:
            pc_path = join(dataset_path, seq_id, 'velodyne')
            file_list.append(
                [join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        file_list = np.concatenate(file_list, axis=0)
        file_list = DataProcessing.shuffle_list(file_list)

        return file_list

    def crop_pc(self, points, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point,
                                       k=self.cfg.num_points)[1][0]
        select_idx = DataProcessing.shuffle_idx(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx
