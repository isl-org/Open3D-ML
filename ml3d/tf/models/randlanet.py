from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from ml3d.datasets.semantickitti import DataProcessing as DP
import tensorflow as tf
import numpy as np
import time
from ..utils import helper_tf_util
from ...datasets.utils.dataprocessing import DataProcessing

# class RandLANet(tf.keras.Model):
#     """docstring for RandLANet"""
#     def __init__(self, cfg):
#         super(RandLANet, self).__init__()
#         self.cfg = cfg
        

class RandLANet(tf.keras.Model):
    def __init__(self, cfg):
        super(RandLANet, self).__init__()
        self.cfg = cfg
        d_feature = cfg.d_feature

        self.fc0 = tf.keras.layers.Dense(d_feature, activation=None)
        self.batch_normalization0 = tf.keras.layers.BatchNormalization(-1, 0.99, 1e-6)
        self.leaky_relu0 = tf.keras.layers.LeakyReLU()
        self.expand_dims0 = helper_tf_util.ExpandDims(axis=2)

        # ###########################Encoder############################
        d_encoder_list = []

        # Encoder
        for i in range(cfg.num_layers):
            name = 'Encoder_layer_' + str(i)
            self.init_dilated_res_block(d_feature, cfg.d_out[i], name)
            d_feature = cfg.d_out[i] * 2
            if i == 0:
                d_encoder_list.append(d_feature)
            d_encoder_list.append(d_feature)

        feature = helper_tf_util.conv2d(True, d_feature, d_feature)
        setattr(self, 'decoder_0', feature)

        # Decoder
        for j in range(cfg.num_layers):
            name = 'Decoder_layer_' + str(j)
            d_in = d_encoder_list[-j - 2] + d_feature
            d_out = d_encoder_list[-j - 2]

            f_decoder_i = helper_tf_util.conv2d_transpose(True, d_in, d_out)
            setattr(self, name, f_decoder_i)
            d_feature = d_encoder_list[-j - 2]

        f_layer_fc1 = helper_tf_util.conv2d(True, d_feature, 64)
        setattr(self, 'fc1', f_layer_fc1)

        f_layer_fc2 = helper_tf_util.conv2d(True, 64, 32)
        setattr(self, 'fc2', f_layer_fc2)

        f_layer_fc3 = helper_tf_util.conv2d(False,
                                            32,
                                            cfg.num_classes,
                                            activation=False)
        setattr(self, 'fc', f_layer_fc3)


    def init_att_pooling(self, d, d_out, name):
        att_activation = tf.keras.layers.Dense(d, activation=None)
        setattr(self, name + 'fc', att_activation)

        f_agg = helper_tf_util.conv2d(True, d, d_out)
        setattr(self, name + 'mlp', f_agg)

    def init_building_block(self, d_in, d_out, name):
        f_pc = helper_tf_util.conv2d(True, d_in)
        setattr(self, name + 'mlp1', f_pc)

        self.init_att_pooling(d_in * 2, d_out // 2, name + 'att_pooling_1')

        f_xyz = helper_tf_util.conv2d(True, d_out // 2)
        setattr(self, name + 'mlp2', f_xyz)

        self.init_att_pooling(d_in * 2, d_out, name + 'att_pooling_2')

    def init_dilated_res_block(self, d_in, d_out, name):
        f_pc = helper_tf_util.conv2d(True, d_out // 2)
        setattr(self, name + 'mlp1', f_pc)

        self.init_building_block(d_out // 2, d_out, name + 'LFA')

        f_pc = helper_tf_util.conv2d(True,
                                     d_out * 2,
                                     activation=False)
        setattr(self, name + 'mlp2', f_pc)

        shortcut = helper_tf_util.conv2d(True,
                                         d_out * 2,
                                         activation=False)
        setattr(self, name + 'shortcut', shortcut)

    def get_loss(self, logits, labels, pre_cal_weights):
        # calculate the weighted cross entropy according to the inverse frequency
        class_weights = tf.convert_to_tensor(pre_cal_weights, dtype=tf.float32)
        one_hot_labels = tf.one_hot(labels, depth=self.config.num_classes)
        weights = tf.reduce_sum(class_weights * one_hot_labels, axis=1)
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
        weighted_losses = unweighted_losses * weights
       
        output_loss = tf.reduce_mean(weighted_losses)
        return output_loss

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        num_neigh = tf.shape(pool_idx)[-1]
        d = feature.get_shape()[-1]
        batch_size = tf.shape(pool_idx)[0]
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = tf.squeeze(feature, axis=2)
        batch_size = tf.shape(interp_idx)[0]
        up_num_points = tf.shape(interp_idx)[1]
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(neighbor_idx)[-1], d])
        return features


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
            select_idx = search_tree.query(center_point, k=num_points)[1][0]

        # select_idx = DataProcessing.shuffle_idx(select_idx)
        random.shuffle(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        if (feat is None):
            select_feat = None
        else:
            select_feat = feat[select_idx]

        select_points = select_points - center_point  # TODO : add noise to center point

        return select_points, select_feat, select_labels, select_idx

    def get_batch_gen(self, dataset):
        cfg = self.cfg

        def gen():
            for i in range(dataset.num_pc):
                data, attr = dataset.read_data(i)
                pick_idx = np.random.choice(len(data['point']), 1)

                pc, feat, label, _ = self.crop_pc(data['point'], data['feat'],
                                                  data['label'],
                                                  data['search_tree'],
                                                  pick_idx)

                label = label[:, 0]

                yield (pc.astype(np.float32), feat.astype(np.float32),
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
            neighbour_idx = tf.py_function(DataProcessing.knn_search,
                                           [pc, pc, cfg.k_n], tf.int32)

            sub_points = pc[:tf.shape(pc)[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:tf.shape(pc)[0] //
                                   cfg.sub_sampling_ratio[i], :]
            up_i = tf.py_function(DataProcessing.knn_search,
                                  [sub_points, pc, 1], tf.int32)
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
