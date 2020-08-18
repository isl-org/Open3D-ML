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

    def crop_pc(self, points, feat, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        num_points = self.cfg.num_points
        center_point = points[pick_idx, :].reshape(1, -1)

        if (points.shape[0] < num_points):
            select_idx = np.array(range(points.shape[0]))
            diff = num_points - points.shape[0]
            select_idx = list(select_idx) + list(
                random.choices(select_idx, k=diff))
            random.shuffle(select_idx)
        else:
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
        
        select_points = select_points - center_point # TODO : add noise to center point

        return select_points, select_feat, select_labels, select_idx

    def get_batch_gen(self, dataset):
        cfg = self.cfg

        def gen():
            for i in range(dataset.num_pc):
                data, attr = dataset.read_data(i)
                pick_idx = np.random.choice(len(data['point']), 1)
                
                pc, feat, label, _ = self.crop_pc(data['point'], data['feat'], data['label'], data['search_tree'], pick_idx)

                label = label[:, 0]

                yield (pc.astype(np.float32),
                       feat.astype(np.float32),
                       label.astype(np.float32))

        gen_func = gen
        gen_types = (tf.float32, tf.float32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None])

        return gen_func, gen_types, gen_shapes


    def transform(self, pc, feat, label):
        cfg = self.cfg

        if (feat is not None):
            features = tf.concat([pc, feat], axis=1)
        else:
            features = pc

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = tf.py_function(DataProcessing.knn_search, [pc, pc, cfg.k_n], tf.int32)

            sub_points = pc[:tf.shape(pc)[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:tf.shape(pc)[0] //
                                    cfg.sub_sampling_ratio[i], :]
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

