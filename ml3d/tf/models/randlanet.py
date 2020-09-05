from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import time
import random
from sklearn.neighbors import KDTree


# use relative import for being compatible with Open3d main repo 
from .base_model import BaseModel
from ...utils import MODEL, helper_tf
from ...datasets.utils import DataProcessing

class RandLANet(BaseModel):
    def __init__(self, 
                name='RandLANet',
                k_n=16,  # KNN,
                num_layers=4,  # Number of layers
                num_points=4096 * 11,  # Number of input points
                num_classes=19,  # Number of valid classes
                ignored_label_inds=[0],
                sub_grid_size=0.06,  # preprocess_parameter
                sub_sampling_ratio=[4, 4, 4, 4],
                num_sub_points=[
                    4096 * 11 // 4, 4096 * 11 // 16, 
                    4096 * 11 // 64, 4096 * 11 // 256
                ],
                dim_input=3,
                dim_feature=8,
                dim_output=[16, 64, 128, 256],
                grid_size=0.06,
                batcher='DefaultBatcher',
                ckpt_path=None,
                **kwargs):

        super().__init__(
            name=name,
            k_n=k_n,
            num_layers=num_layers,
            num_points=num_points,
            num_classes=num_classes,
            ignored_label_inds=ignored_label_inds,
            sub_grid_size=sub_grid_size,
            sub_sampling_ratio=sub_sampling_ratio,
            num_sub_points=num_sub_points,
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
            -1, 0.99, 1e-6)
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

        f_layer_fc3 = helper_tf.conv2d(False,
                                       cfg.num_classes,
                                       activation=False)
        setattr(self, 'fc', f_layer_fc3)


    def init_att_pooling(self, d, dim_output, name):
        att_activation = tf.keras.layers.Dense(d, activation=None)
        setattr(self, name + 'fc', att_activation)

        f_agg = helper_tf.conv2d(True, dim_output)
        setattr(self, name + 'mlp', f_agg)

    def init_building_block(self, dim_input, dim_output, name):
        f_pc = helper_tf.conv2d(True, dim_input)

        setattr(self, name + 'bdmlp1', f_pc)

        self.init_att_pooling(dim_input * 2, dim_output // 2, name + 'att_pooling_1')

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
        m_conv2d = getattr(self, name + 'bdmlp1')

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

    def forward_dilated_res_block(self, feature, xyz, neigh_idx, dim_output, name):
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
            f_interp_i = self.nearest_interpolation(feature,
                                                    interp_idx[-j - 1])
            name = 'Decoder_layer_' + str(j)

            m_transposeconv2d = getattr(self, name)
            concat_feature = tf.concat([f_encoder_list[-j - 2], f_interp_i],
                                       axis=3)
            f_decoder_i = m_transposeconv2d(concat_feature, training=self.training)

            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)

        m_conv2d = getattr(self, 'fc1')
        f_layer_fc1 = m_conv2d(f_decoder_list[-1], training=self.training)

        m_conv2d = getattr(self, 'fc2')
        f_layer_fc2 = m_conv2d(f_layer_fc1, training=self.training)

        m_dropout = getattr(self, 'dropout1')
        f_layer_drop = m_dropout(f_layer_fc2, training=self.training)

        m_conv2d = getattr(self, 'fc')
        f_layer_fc3 = m_conv2d(f_layer_drop, training=self.training)

        f_out = tf.squeeze(f_layer_fc3, [2])

        return f_out

    def get_optimizer(self, cfg_pipeline):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            cfg_pipeline.adam_lr, decay_steps=100000, decay_rate=cfg_pipeline.scheduler_gamma)
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
        interpolatedim_features = tf.expand_dims(interpolatedim_features, axis=2)
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

        points = data['point'][:, 0:3]
        labels = data['label']
        split = attr['split']

        if 'feat' not in data.keys() or data['feat'] is None:
            feat = points
        else:
            feat = np.array(data['feat'], dtype=np.float32)
            feat = np.concatenate([points, feat], axis=1)

        data = dict()

        if (feat is None):
            sub_points, sub_labels = DataProcessing.grid_subsampling(
                points, labels=labels, grid_size=cfg.grid_size)

        else:
            sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
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

MODEL._register_module(RandLANet, 'tf')

