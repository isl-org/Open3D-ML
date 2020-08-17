from os import makedirs
from os.path import exists
import time
import tensorflow as tf
import numpy as np
import random
from os.path import exists, join, isfile, dirname, abspath, split
import sys
from pathlib import Path
from sklearn.neighbors import KDTree

from ...datasets.utils.dataprocessing import DataProcessing


class RandLANet:

    def __init__(self, cfg):
        # Model parameters
        self.cfg = cfg

    def get_batch_gen(self, dataset):
        def gen():
            for i in range(dataset.num_pc):
                yield i
        return gen, tf.int64, []


    def transform(pc, feat, label):
        num_layers = 5
        k_n = 16
        sub_sampling_ratio = [4, 4, 4, 4, 2]

        if (feat is not None):
            features = tf.concat([pc, feat], axis=1)
        else:
            features = pc

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(num_layers):
            neighbour_idx = tf.py_function(DataProcessing.knn_search, [pc, pc, k_n], tf.int32)

            sub_points = pc[:tf.shape(pc)[0] // sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:tf.shape(pc)[0] //
                                    sub_sampling_ratio[i], :]
            up_i = tf.py_function(DataProcessing.knn_search, [sub_points, pc, 1], tf.int32)
            input_points.append(pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, label]

        return input_list

    def preprocess(self, data, attr):
        cfg = self.cfg
        if 'feat' not in data.keys():
            data['feat'] = None

        points = data['point'][:, 0:3]
        feat = data['feat'][:, 0:3]
        labels = data['label']
        split = attr['split']

        if (feat is None):
            sub_feat = None

        data = dict()

        if (feat is None):
            sub_points, sub_labels = DataProcessing.grid_sub_sampling(
                points, labels=labels, grid_size=cfg.grid_size)

        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_sub_sampling(
                points, features=feat, labels=labels, grid_size=cfg.grid_size)

        search_tree = KDTree(sub_points)

        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['search_tree'] = search_tree

        if split != "training":
            proj_inds = np.squeeze(
                search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds

        return data

    def crop_pc(self, points, feat, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        num_points = 65536
        if (points.shape[0] < num_points):
            select_idx = np.array(range(points.shape[0]))
            diff = num_points - points.shape[0]
            select_idx = list(select_idx) + list(
                random.choices(select_idx, k=diff))
            random.shuffle(select_idx)
        else:
            center_point = points[pick_idx, :].reshape(1, -1)
            select_idx = search_tree.query(center_point,
                                            k=num_points)[1][0]

        # select_idx = DataProcessing.shuffle_idx(select_idx)
        random.shuffle(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        if (feat is None):
            select_feat = None
        else:
            select_feat = feat[select_idx]
        return select_points, select_feat, select_labels, select_idx

