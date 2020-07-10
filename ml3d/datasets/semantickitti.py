import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler
import torch

import utils.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import utils.nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors

import yaml


BASE_DIR = './'
#BASE_DIR = dirname(abspath(__file__))

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




# yaml
# network
# dataset
class ConfigSemanticKITTI:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 4096 * 11  # Number of input points
    num_classes = 19  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    d_in      = 3
    d_feature = 8 

    batch_size = 1  # batch_size during training
    val_batch_size = 1  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    main_log_dir = './logs'
    model_name = 'RandLANet'
    logs_dir = join(main_log_dir, model_name)
    ckpt_path = './ml3d/torch/checkpoint/randlanet_semantickitti.pth'

    # test
    test_split_number   = 11
    test_result_folder = './test'

    # inference
    grid_size = 0.06

    # training
    training_split = ['00', '01', '02', '03', '04', '05', 
                        '06', '07', '09', '10']
    save_ckpt_freq = 20

    adam_lr         = 1e-2
    scheduler_gamma = 0.95
    sampling_type = 'active_learning'

    class_weights = [ 55437630, 320797, 541736, 2578735, 3274484, 552662, 
                        184064, 78858, 240942562, 17294618, 170599734, 
                        6369672, 230413074, 101130274, 476491114,9833174, 
                        129609852, 4506626, 1168181]
   


class DataProcessing:
    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def get_file_list(dataset_path, test_scan_num):
        seq_list = np.sort(os.listdir(dataset_path))

        train_file_list = []
        test_file_list = []
        val_file_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            if seq_id == '08':
                val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
                if seq_id == test_scan_num:
                    test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif int(seq_id) >= 11 and seq_id == test_scan_num:
                test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        train_file_list = np.concatenate(train_file_list, axis=0)
        val_file_list = np.concatenate(val_file_list, axis=0)
        test_file_list = np.concatenate(test_file_list, axis=0)
        return train_file_list, val_file_list, test_file_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        """
        CPP wrapper for a grid sub_sampling (method = barycenter for points and features
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param grid_size: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: sub_sampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(dataset_name):
        # pre-calculate the number of points in each category
        num_per_class = []
        if dataset_name is 'S3DIS':
            num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                                      650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
        elif dataset_name is 'Semantic3D':
            num_per_class = np.array([5181602, 5012952, 6830086, 1311528, 10476365, 946982, 334860, 269353],
                                     dtype=np.int32)
        elif dataset_name is 'SemanticKITTI':
            num_per_class = np.array([55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                                      240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114,
                                      9833174, 129609852, 4506626, 1168181])
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)



class SimpleSampler(IterableDataset):
    def __init__(self, dataset, split='training'):
        cfg         = dataset.cfg
        path_list   = dataset.get_split_list(split)

        test = 0
        if split == 'training':
            num_per_epoch = int(len(path_list) / cfg.batch_size) 
            dataset.train_list = path_list
        elif split == 'validation':
            num_per_epoch = int(len(path_list) / cfg.val_batch_size) 
            dataset.val_list = path_list
        elif split == 'test':
            num_per_epoch = int(len(path_list) / cfg.val_batch_size) * 4
            dataset.test_list = path_list
            for test_file_name in path_list:
                points = np.load(test_file_name)
                dataset.possibility += [np.random.rand(points.shape[0]) * 1e-3]
                dataset.min_possibility += [float(np.min(dataset.possibility[-1]))]
                test = test + 1
                print(test_file_name)
                if test == 8:
                    break

        self.num_per_epoch  = num_per_epoch
        self.path_list      = path_list
        self.split          = split
        self.dataset        = dataset

        
    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.num_per_epoch 

    def spatially_regular_gen(self):
        for i in range(self.num_per_epoch):
            if self.split != 'test':
                is_test     = False
                cloud_ind   = i
                pc_path     = self.path_list[cloud_ind]
                pc, tree, labels = self.dataset.get_data(pc_path, is_test)
                pick_idx    = np.random.choice(len(pc), 1)
                selected_pc, selected_labels, selected_idx = \
                    self.dataset.crop_pc(pc, labels, tree, pick_idx)
            else:
                is_test     = True
                cloud_ind   = int(np.argmin(self.dataset.min_possibility))
                pc_path     = self.path_list[cloud_ind]
                pc, tree, labels = self.dataset.get_data(pc_path, is_test)
                pick_idx    = np.argmin(self.dataset.possibility[cloud_ind])
                selected_pc, selected_labels, selected_idx = \
                    self.dataset.crop_pc(pc, labels, tree, pick_idx)
 
            if self.split == 'test':
                # update the possibility of the selected pc
                dists = np.sum(np.square((selected_pc - pc[pick_idx]).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.dataset.possibility[cloud_ind][selected_idx] += delta
                self.dataset.min_possibility[cloud_ind] = np.min(self.dataset.possibility[cloud_ind])

            yield (selected_pc.astype(np.float32),
                    selected_labels.astype(np.int64),
                    selected_idx.astype(np.int64),
                    np.array([cloud_ind], dtype=np.int64))
        


class SemanticKITTI:
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = 'SemanticKITTI'
        self.label_to_names = {0: 'unlabeled',
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
                               19: 'traffic-sign'}
        self.num_classes = len(self.label_to_names)
        
        self.possibility = []
        self.min_possibility = []

        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([0])
        self.cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]

    def get_sampler (self, split):
        return SimpleSampler(self, split=split)

    def preprocess(self, batch_data, device):
        cfg             = self.cfg
        batch_pc        = batch_data[0]
        batch_label     = batch_data[1]
        batch_pc_idx    = batch_data[2]
        batch_cloud_idx = batch_data[3]

        features         = batch_pc
        input_points     = []
        input_neighbors  = []
        input_pools      = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DataProcessing.knn_search(batch_pc, batch_pc, cfg.k_n)
            
            sub_points = batch_pc[:, :batch_pc.size(1) // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.size(1) // cfg.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        inputs = dict()
        #print(features)
        inputs['xyz']           = [arr.to(device) 
                                    for arr in input_points]
        inputs['neigh_idx']     = [torch.from_numpy(arr).to(torch.int64).to(device) 
                                    for arr in input_neighbors]
        inputs['sub_idx']       = [torch.from_numpy(arr).to(torch.int64).to(device) 
                                    for arr in input_pools]
        inputs['interp_idx']    = [torch.from_numpy(arr).to(torch.int64).to(device) 
                                    for arr in input_up_samples]
        inputs['features']      = features.to(device)
        inputs['input_inds']    = batch_pc_idx
        inputs['cloud_inds']    = batch_cloud_idx

        #print(input_points)   
        #print(input_neighbors)        
        #features    = [torch.from_numpy(arr) for arr in inputs['features']]

        return inputs

    def save_test_result(self, test_probs):
        cfg = self.cfg
        
        test_path = join(cfg.test_result_folder, 'sequences')
        makedirs(test_path) if not exists(test_path) else None
        save_path = join(test_path, str(cfg.test_split_number), 
                                'predictions')
        makedirs(save_path) if not exists(save_path) else None

        test_scan_name = str(self.cfg.test_split_number)

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


    def get_data(self, file_path, is_test=False):
        seq_id          = file_path.split('/')[-3]
        frame_id        = file_path.split('/')[-1][:-4]
        kd_tree_path    = join(self.cfg.dataset_path, seq_id, 
                                'KDTree', frame_id + '.pkl')
        # Read pkl with search tree
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)

        # Load labels
        if is_test:
            labels      = np.zeros(np.shape(points)[0], dtype=np.uint8)
        else:
            label_path  = join(self.cfg.dataset_path, seq_id, 'labels', 
                                frame_id + '.npy')
            labels      = np.squeeze(np.load(label_path))
        return points, search_tree, labels


    def get_split_list(self, split):
        cfg             = self.cfg
        dataset_path    = cfg.dataset_path
        file_list       = []

        if split == 'training':
            seq_list = cfg.training_split
        elif split == 'test':
            seq_list = [str(cfg.test_split_number)]
        elif split == 'validation':
            pass

        for seq_id in seq_list:
            pc_path = join(dataset_path, seq_id, 'velodyne')
            file_list.append([join(pc_path, f) for f in 
                                np.sort(os.listdir(pc_path))])

        file_list = np.concatenate(file_list, axis=0)
        file_list = DataProcessing.shuffle_list(file_list)
        
        return file_list


    def crop_pc(self, points, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, k=self.cfg.num_points)[1][0]
        select_idx = DataProcessing.shuffle_idx(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx
