from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import time
import random
from tqdm import tqdm
from sklearn.neighbors import KDTree

# use relative import for being compatible with Open3d main repo
from .base_model import BaseModel
from ..utils import helper_tf
from ...utils import MODEL
from ...datasets.utils import (DataProcessing, trans_normalize, trans_augment,
                               trans_crop_pc)


class RandLANet(BaseModel):

    def __init__(
            self,
            name='RandLANet',
            k_n=16,  # KNN,
            num_layers=4,  # Number of layers
            num_points=4096 * 11,  # Number of input points
            num_classes=19,  # Number of valid classes
            ignored_label_inds=[0],
            sub_sampling_ratio=[4, 4, 4, 4],
            dim_input=3,
            dim_feature=8,
            dim_output=[16, 64, 128, 256],
            grid_size=0.06,
            batcher='DefaultBatcher',
            ckpt_path=None,
            **kwargs):

        super().__init__(name=name,
                         k_n=k_n,
                         num_layers=num_layers,
                         num_points=num_points,
                         num_classes=num_classes,
                         ignored_label_inds=ignored_label_inds,
                         sub_sampling_ratio=sub_sampling_ratio,
                         dim_input=dim_input,
                         dim_feature=dim_feature,
                         dim_output=dim_output,
                         grid_size=grid_size,
                         batcher=batcher,
                         ckpt_path=ckpt_path,
                         **kwargs)

        cfg = self.cfg

        dim_feature = cfg.dim_feature
        self.fc0 = tf.keras.layers.Dense(dim_feature, activation=None)
        self.batch_normalization = tf.keras.layers.BatchNormalization(
            -1, momentum=0.99, epsilon=1e-6)
        self.leaky_relu0 = tf.keras.layers.LeakyReLU()

        # ###########################Encoder############################
        d_encoder_list = []

        # Encoder
        for i in range(cfg.num_layers):
            name = 'Encoder_layer_' + str(i)
            self.init_dilated_res_block(dim_feature, cfg.dim_output[i], name)
            dim_feature = cfg.dim_output[i] * 2
            if i == 0:
                d_encoder_list.append(dim_feature)
            d_encoder_list.append(dim_feature)

        feature = helper_tf.conv2d(True, dim_feature)
        setattr(self, 'decoder_0', feature)

        # Decoder
        for j in range(cfg.num_layers):
            name = 'Decoder_layer_' + str(j)
            dim_input = d_encoder_list[-j - 2] + dim_feature
            dim_output = d_encoder_list[-j - 2]

            f_decoder_i = helper_tf.conv2d_transpose(True, dim_output)
            setattr(self, name, f_decoder_i)
            dim_feature = d_encoder_list[-j - 2]

        f_layer_fc1 = helper_tf.conv2d(True, 64)
        setattr(self, 'fc1', f_layer_fc1)

        f_layer_fc2 = helper_tf.conv2d(True, 32)
        setattr(self, 'fc2', f_layer_fc2)

        f_dropout = tf.keras.layers.Dropout(0.5)
        setattr(self, 'dropout1', f_dropout)

        f_layer_fc3 = helper_tf.conv2d(False, cfg.num_classes, activation=False)
        setattr(self, 'fc', f_layer_fc3)

    def init_att_pooling(self, d, dim_output, name):
        att_activation = tf.keras.layers.Dense(d, activation=None)
        setattr(self, name + 'fc', att_activation)

        f_agg = helper_tf.conv2d(True, dim_output)
        setattr(self, name + 'mlp', f_agg)

    def init_building_block(self, dim_input, dim_output, name):
        f_pc = helper_tf.conv2d(True, dim_input)

        setattr(self, name + 'mlp1', f_pc)

        self.init_att_pooling(dim_input * 2, dim_output // 2,
                              name + 'att_pooling_1')

        f_xyz = helper_tf.conv2d(True, dim_output // 2)
        setattr(self, name + 'mlp2', f_xyz)

        self.init_att_pooling(dim_input * 2, dim_output, name + 'att_pooling_2')

    def init_dilated_res_block(self, dim_input, dim_output, name):
        f_pc = helper_tf.conv2d(True, dim_output // 2)
        setattr(self, name + 'mlp1', f_pc)

        self.init_building_block(dim_output // 2, dim_output, name + 'LFA')

        f_pc = helper_tf.conv2d(True, dim_output * 2, activation=False)
        setattr(self, name + 'mlp2', f_pc)

        shortcut = helper_tf.conv2d(True, dim_output * 2, activation=False)
        setattr(self, name + 'shortcut', shortcut)

    def forward_gather_neighbour(self, pc, neighbor_idx):
        # pc:           BxNxd
        # neighbor_idx: BxNxK
        B, N, K = neighbor_idx.shape
        d = pc.shape[2]

        index_input = tf.reshape(neighbor_idx, shape=[-1, N * K])

        features = tf.gather(pc, index_input, axis=1, batch_dims=1)

        features = tf.reshape(features, [-1, N, K, d])

        return features

    def forward_att_pooling(self, feature_set, name):
        # feature_set: BxNxKxd
        batch_size = feature_set.shape[0]
        num_points = feature_set.shape[1]
        num_neigh = feature_set.shape[2]
        d = feature_set.shape[3]

        f_reshaped = tf.reshape(feature_set, shape=[-1, num_neigh, d])

        m_dense = getattr(self, name + 'fc')
        att_activation = m_dense(f_reshaped)
        att_scores = tf.nn.softmax(att_activation, axis=1)

        # print("att_scores = ", att_scores.shape)
        f_agg = f_reshaped * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [-1, num_points, 1, d])

        m_conv2d = getattr(self, name + 'mlp')
        f_agg = m_conv2d(f_agg, training=self.training)

        return f_agg

    def forward_relative_pos_encoding(self, xyz, neigh_idx):
        B, N, K = neigh_idx.shape
        neighbor_xyz = self.forward_gather_neighbour(xyz, neigh_idx)

        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2),
                           [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(
            tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat(
            [relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature

    def forward_building_block(self, xyz, feature, neigh_idx, name):
        f_xyz = self.forward_relative_pos_encoding(xyz, neigh_idx)
        m_conv2d = getattr(self, name + 'mlp1')

        f_xyz = m_conv2d(f_xyz, training=self.training)

        f_neighbours = self.forward_gather_neighbour(
            tf.squeeze(feature, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)

        f_pc_agg = self.forward_att_pooling(f_concat, name + 'att_pooling_1')

        m_conv2d = getattr(self, name + 'mlp2')
        f_xyz = m_conv2d(f_xyz, training=self.training)

        f_neighbours = self.forward_gather_neighbour(
            tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_concat = tf.concat([f_neighbours, f_xyz], axis=-1)
        f_pc_agg = self.forward_att_pooling(f_concat, name + 'att_pooling_2')

        return f_pc_agg

    def forward_dilated_res_block(self, feature, xyz, neigh_idx, dim_output,
                                  name):
        m_conv2d = getattr(self, name + 'mlp1')
        f_pc = m_conv2d(feature, training=self.training)

        f_pc = self.forward_building_block(xyz, f_pc, neigh_idx, name + 'LFA')

        m_conv2d = getattr(self, name + 'mlp2')
        f_pc = m_conv2d(f_pc, training=self.training)

        m_conv2d = getattr(self, name + 'shortcut')
        shortcut = m_conv2d(feature, training=self.training)

        result = tf.nn.leaky_relu(f_pc + shortcut)
        return result

    def call(self, inputs, training=True):
        self.training = training
        num_layers = self.cfg.num_layers
        xyz = inputs[:num_layers]
        neigh_idx = inputs[num_layers:2 * num_layers]
        sub_idx = inputs[2 * num_layers:3 * num_layers]
        interp_idx = inputs[3 * num_layers:4 * num_layers]
        feature = inputs[4 * num_layers]

        m_dense = getattr(self, 'fc0')
        feature = m_dense(feature, training=self.training)

        m_bn = getattr(self, 'batch_normalization')
        feature = m_bn(feature, training=self.training)

        feature = tf.nn.leaky_relu(feature)
        feature = tf.expand_dims(feature, axis=2)

        # B N 1 d
        # Encoder
        f_encoder_list = []
        for i in range(self.cfg.num_layers):
            name = 'Encoder_layer_' + str(i)
            f_encoder_i = self.forward_dilated_res_block(
                feature, xyz[i], neigh_idx[i], self.cfg.dim_output[i], name)
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)

        m_conv2d = getattr(self, 'decoder_0')
        feature = m_conv2d(f_encoder_list[-1], training=self.training)

        # Decoder
        f_decoder_list = []
        for j in range(self.cfg.num_layers):
            f_interp_i = self.nearest_interpolation(feature, interp_idx[-j - 1])
            name = 'Decoder_layer_' + str(j)

            m_transposeconv2d = getattr(self, name)
            concat_feature = tf.concat([f_encoder_list[-j - 2], f_interp_i],
                                       axis=3)
            f_decoder_i = m_transposeconv2d(concat_feature,
                                            training=self.training)

            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)

        m_conv2d = getattr(self, 'fc1')
        f_layer_fc1 = m_conv2d(f_decoder_list[-1], training=self.training)

        m_conv2d = getattr(self, 'fc2')
        f_layer_fc2 = m_conv2d(f_layer_fc1, training=self.training)

        self.test_hidden = f_layer_fc2

        m_dropout = getattr(self, 'dropout1')
        f_layer_drop = m_dropout(f_layer_fc2, training=self.training)

        m_conv2d = getattr(self, 'fc')
        f_layer_fc3 = m_conv2d(f_layer_drop, training=self.training)

        f_out = tf.squeeze(f_layer_fc3, [2])
        # f_out = tf.nn.softmax(f_out)

        return f_out

    def get_optimizer(self, cfg_pipeline):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            cfg_pipeline.adam_lr,
            decay_steps=100000,
            decay_rate=cfg_pipeline.scheduler_gamma)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        return optimizer

    def get_loss(self, Loss, results, inputs):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """
        cfg = self.cfg
        labels = inputs[-1]

        scores, labels = Loss.filter_valid_label(results, labels)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

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

        pool_features = tf.gather(feature, pool_idx, axis=1, batch_dims=1)

        pool_features = tf.reshape(pool_features,
                                   [batch_size, -1, num_neigh, d])

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

        interpolatedim_features = tf.gather(feature,
                                            interp_idx,
                                            axis=1,
                                            batch_dims=1)
        interpolatedim_features = tf.expand_dims(interpolatedim_features,
                                                 axis=2)
        return interpolatedim_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        # gather the coordinates or features of neighboring points
        batch_size = tf.shape(pc)[0]
        num_points = tf.shape(pc)[1]
        d = pc.get_shape()[2].value
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.batch_gather(pc, index_input)
        features = tf.reshape(
            features, [batch_size, num_points,
                       tf.shape(neighbor_idx)[-1], d])
        return features

    def get_batch_gen(self, dataset, steps_per_epoch=None, batch_size=1):
        cfg = self.cfg

        def gen():
            n_iters = dataset.num_pc if steps_per_epoch is None else steps_per_epoch * batch_size
            for i in range(n_iters):
                data, attr = dataset.read_data(i % dataset.num_pc)

                pc = data['point'].copy()
                label = data['label'].copy()
                feat = data['feat'].copy() if data['feat'] is not None else None
                tree = data['search_tree']

                pick_idx = np.random.choice(len(pc), 1)
                center_point = pc[pick_idx, :].reshape(1, -1)
                pc, feat, label, _ = trans_crop_pc(pc, feat, label, tree,
                                                   pick_idx,
                                                   self.cfg.num_points)

                if not cfg.get('recentering', True):
                    pc = pc + center_point

                t_normalize = cfg.get('t_normalize', None)
                pc, feat = trans_normalize(pc, feat, t_normalize)

                if attr['split'] in ['training', 'train']:
                    t_augment = cfg.get('t_augment', None)
                    pc = trans_augment(pc, t_augment)

                if feat is None:
                    feat = pc.copy()
                else:
                    feat = np.concatenate([pc, feat], axis=1)
                assert self.cfg.dim_input == feat.shape[
                    1], "Wrong feature dimension, please update dim_input(3 + feature_dimension) in config"

                yield (pc.astype(np.float32), feat.astype(np.float32),
                       label.astype(np.float32))

        gen_func = gen
        gen_types = (tf.float32, tf.float32, tf.int32)
        gen_shapes = ([None, 3], [None, cfg.dim_input], [None])

        return gen_func, gen_types, gen_shapes

    def transform_inference(self, data, min_posbility_idx):
        cfg = self.cfg
        inputs = dict()

        pc = data['point'].copy()
        label = data['label'].copy()
        feat = data['feat'].copy() if data['feat'] is not None else None
        tree = data['search_tree']

        pick_idx = min_posbility_idx
        center_point = pc[pick_idx, :].reshape(1, -1)

        pc, feat, label, selected_idx = trans_crop_pc(pc, feat, label, tree,
                                                      pick_idx,
                                                      self.cfg.num_points)

        dists = np.sum(np.square(pc.astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[selected_idx] += delta
        inputs['point_inds'] = selected_idx

        if not cfg.get('recentering', True):
            pc = pc + center_point

        t_normalize = cfg.get('t_normalize', None)
        pc, feat = trans_normalize(pc, feat, t_normalize)

        if feat is None:
            feat = pc.copy()
        else:
            feat = np.concatenate([pc, feat], axis=1)

        assert self.cfg.dim_input == feat.shape[
            1], "Wrong feature dimension, please update dim_input(3 + feature_dimension) in config"

        features = feat
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DataProcessing.knn_search(pc, pc, cfg.k_n)

            sub_points = pc[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, pc, 1)
            input_points.append(pc)
            input_neighbors.append(neighbour_idx.astype(np.int64))
            input_pools.append(pool_i.astype(np.int64))
            input_up_samples.append(up_i.astype(np.int64))
            pc = sub_points

        inputs['xyz'] = input_points
        inputs['neigh_idx'] = input_neighbors
        inputs['sub_idx'] = input_pools
        inputs['interp_idx'] = input_up_samples
        inputs['features'] = features

        inputs['labels'] = label.astype(np.int64)

        return inputs

    def transform(self, pc, feat, label):
        cfg = self.cfg

        pc = pc
        feat = feat

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = tf.numpy_function(DataProcessing.knn_search,
                                              [pc, pc, cfg.k_n], tf.int32)

            sub_points = pc[:tf.shape(pc)[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:tf.shape(pc)[0] //
                                   cfg.sub_sampling_ratio[i], :]
            up_i = tf.numpy_function(DataProcessing.knn_search,
                                     [sub_points, pc, 1], tf.int32)
            input_points.append(pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [feat, label]

        return input_list

    def inference_begin(self, data):
        self.test_smooth = 0.95
        attr = {'split': 'test'}
        self.inference_data = self.preprocess(data, attr)
        num_points = self.inference_data['search_tree'].data.shape[0]
        self.possibility = np.random.rand(num_points) * 1e-3
        self.test_probs = np.zeros(shape=[num_points, self.cfg.num_classes],
                                   dtype=np.float16)
        self.pbar = tqdm(total=self.possibility.shape[0])
        self.pbar_update = 0

    def inference_preprocess(self):
        min_posbility_idx = np.argmin(self.possibility)
        data = self.transform_inference(self.inference_data, min_posbility_idx)
        inputs = {'data': data, 'attr': []}
        # inputs = self.batcher.collate_fn([inputs])
        self.inference_input = inputs

        flat_inputs = data['xyz'] + data['neigh_idx'] + data['sub_idx'] + data[
            'interp_idx']
        flat_inputs += [data['features'], data['labels']]

        for i in range(len(flat_inputs)):
            flat_inputs[i] = np.expand_dims(flat_inputs[i], 0)

        return flat_inputs

    def inference_end(self, results):
        inputs = self.inference_input
        results = tf.reshape(results, (-1, self.cfg.num_classes))
        results = tf.nn.softmax(results, axis=-1)
        results = results.cpu().numpy()

        probs = np.reshape(results, [-1, self.cfg.num_classes])
        inds = inputs['data']['point_inds']
        self.test_probs[inds] = self.test_smooth * self.test_probs[inds] + (
            1 - self.test_smooth) * probs

        self.pbar.update(self.possibility[self.possibility > 0.5].shape[0] -
                         self.pbar_update)
        self.pbar_update = self.possibility[self.possibility > 0.5].shape[0]

        if np.min(self.possibility) > 0.5:
            self.pbar.close()
            reproj_inds = self.inference_data['proj_inds']
            self.test_probs = self.test_probs[reproj_inds]
            inference_result = {
                'predict_labels': np.argmax(self.test_probs, 1),
                'predict_scores': self.test_probs
            }
            self.inference_result = inference_result
            return True
        else:
            return False

    def preprocess(self, data, attr):
        cfg = self.cfg

        points = data['point'][:, 0:3]

        if 'label' not in data.keys() or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data.keys() or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        split = attr['split']

        data = dict()

        if (feat is None):
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=cfg.grid_size)
            sub_feat = None

        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                points, features=feat, labels=labels, grid_size=cfg.grid_size)

        search_tree = KDTree(sub_points)

        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['search_tree'] = search_tree

        if split in ["test", "testing"]:
            proj_inds = np.squeeze(
                search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds

        return data


MODEL._register_module(RandLANet, 'tf')
