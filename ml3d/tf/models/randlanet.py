from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from ml3d.datasets.semantickitti import DataProcessing as DP
import tensorflow as tf
import numpy as np
from ..utils import helper_tf_util
import time

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
