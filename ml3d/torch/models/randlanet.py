import torch
import torch.nn as nn
import numpy as np
import random
import time
from tqdm import tqdm

from pathlib import Path
from sklearn.neighbors import KDTree
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler

# use relative import for being compatible with Open3d main repo
from .base_model import BaseModel
from ..utils import helper_torch
from ..dataloaders import DefaultBatcher
from ..modules.losses import filter_valid_label
from ...datasets.utils import (DataProcessing, trans_normalize, trans_augment,
                               trans_crop_pc)
from ...utils import MODEL


class RandLANet(BaseModel):
    """Class defining RandLANet.

    A model for Semantic Segmentation.
    """

    def __init__(
            self,
            name='RandLANet',
            k_n=16,  # KNN,
            num_layers=4,  # Number of layers
            num_points=4096 * 11,  # Number of input points
            num_classes=19,  # Number of valid classes
            ignored_label_inds=[0],
            sub_sampling_ratio=[4, 4, 4, 4],
            dim_input=3,  # 3 + feature_dimension.
            dim_feature=8,
            dim_output=[16, 64, 128, 256],
            grid_size=0.06,
            batcher='DefaultBatcher',
            ckpt_path=None,
            weight_decay=0.0,
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
                         weight_decay=weight_decay,
                         **kwargs)
        cfg = self.cfg

        dim_feature = cfg.dim_feature
        self.fc0 = nn.Linear(cfg.dim_input, dim_feature)
        self.batch_normalization = nn.BatchNorm2d(dim_feature,
                                                  eps=1e-6,
                                                  momentum=0.01)

        d_encoder_list = []

        # Encoder
        for i in range(cfg.num_layers):
            name = 'Encoder_layer_' + str(i)
            self.init_dilated_res_block(dim_feature, cfg.dim_output[i], name)
            dim_feature = cfg.dim_output[i] * 2
            if i == 0:
                d_encoder_list.append(dim_feature)
            d_encoder_list.append(dim_feature)

        feature = helper_torch.conv2d(True, dim_feature, dim_feature)
        setattr(self, 'decoder_0', feature)

        # Decoder
        for j in range(cfg.num_layers):
            name = 'Decoder_layer_' + str(j)
            dim_input = d_encoder_list[-j - 2] + dim_feature
            dim_output = d_encoder_list[-j - 2]

            f_decoder_i = helper_torch.conv2d_transpose(True, dim_input,
                                                        dim_output)
            setattr(self, name, f_decoder_i)
            dim_feature = d_encoder_list[-j - 2]

        f_layer_fc1 = helper_torch.conv2d(True, dim_feature, 64)
        setattr(self, 'fc1', f_layer_fc1)

        f_layer_fc2 = helper_torch.conv2d(True, 64, 32)
        setattr(self, 'fc2', f_layer_fc2)

        f_layer_fc3 = helper_torch.conv2d(False,
                                          32,
                                          cfg.num_classes,
                                          activation=False)
        setattr(self, 'fc', f_layer_fc3)

        self.m_dropout = nn.Dropout(0.5)

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=cfg_pipeline.adam_lr,
                                     weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler

    def get_loss(self, Loss, results, inputs, device):
        """Runs the loss on outputs of the model.

        Args:
            outputs: logits
            labels: labels

        Returns:
             loss
        """
        cfg = self.cfg
        labels = inputs['data']['labels']

        scores, labels = filter_valid_label(results, labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def update_probs(self, inputs, results, test_probs, test_labels):
        self.test_smooth = 0.95

        for b in range(results.size()[0]):

            result = torch.reshape(results[b], (-1, self.cfg.num_classes))
            probs = torch.nn.functional.softmax(result, dim=-1)
            probs = probs.cpu().data.numpy()
            labels = np.argmax(probs, 1)
            inds = inputs['data']['point_inds'][b]

            test_probs[inds] = self.test_smooth * test_probs[inds] + (
                1 - self.test_smooth) * probs
            test_labels[inds] = labels

        return test_probs, test_labels

    def transform(self, data, attr, min_possibility_idx=None):
        cfg = self.cfg
        inputs = dict()

        pc = data['point'].copy()
        label = data['label'].copy()
        feat = data['feat'].copy() if data['feat'] is not None else None
        tree = data['search_tree']

        pc, selected_idxs, center_point = self.trans_point_sampler(
            pc=pc,
            feat=feat,
            label=label,
            search_tree=tree,
            num_points=self.cfg.num_points)

        label = label[selected_idxs]
        if (feat is None):
            feat = None
        else:
            feat = feat[selected_idxs]

        t_normalize = cfg.get('t_normalize', {})
        pc, feat = trans_normalize(pc, feat, t_normalize)

        if attr['split'] in ['training', 'train']:
            t_augment = cfg.get('t_augment', None)
            pc = trans_augment(pc, t_augment)

        if feat is None:
            feat = pc.copy()
        else:
            feat = np.concatenate([pc, feat], axis=1)

        assert cfg.dim_input == feat.shape[
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
        inputs['point_inds'] = selected_idxs

        inputs['labels'] = label.astype(np.int64)
        return inputs

    def inference_begin(self, data):
        self.test_smooth = 0.95
        attr = {'split': 'test'}
        self.inference_ori_data = data
        self.inference_data = self.preprocess(data, attr)
        self.inference_proj_inds = self.inference_data['proj_inds']
        num_points = self.inference_data['search_tree'].data.shape[0]
        self.possibility = np.random.rand(num_points) * 1e-3
        self.test_probs = np.zeros(shape=[num_points, self.cfg.num_classes],
                                   dtype=np.float16)
        self.pbar = tqdm(total=self.possibility.shape[0])
        self.pbar_update = 0
        self.batcher = DefaultBatcher()

    def inference_preprocess(self):
        min_possibility_idx = np.argmin(self.possibility)
        attr = {'split': 'test'}
        data = self.transform(self.inference_data, attr, min_possibility_idx)
        inputs = {'data': data, 'attr': attr}
        inputs = self.batcher.collate_fn([inputs])
        self.inference_input = inputs

        return inputs

    def inference_end(self, inputs, results):

        results = torch.reshape(results, (-1, self.cfg.num_classes))
        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()
        probs = np.reshape(results, [-1, self.cfg.num_classes])

        pred_l = np.argmax(probs, 1)

        inds = inputs['data']['point_inds']
        self.test_probs[inds] = self.test_smooth * self.test_probs[inds] + (
            1 - self.test_smooth) * probs

        self.pbar.update(self.possibility[self.possibility > 0.5].shape[0] -
                         self.pbar_update)
        self.pbar_update = self.possibility[self.possibility > 0.5].shape[0]
        if np.min(self.possibility) > 0.5:
            self.pbar.close()
            pred_labels = np.argmax(self.test_probs, 1)

            pred_labels = pred_labels[self.inference_proj_inds]
            test_probs = self.test_probs[self.inference_proj_inds]
            inference_result = {
                'predict_labels': pred_labels,
                'predict_scores': test_probs
            }
            data = self.inference_ori_data
            acc = (pred_labels == data['label'] - 1).mean()

            self.inference_result = inference_result
            return True
        else:
            return False

    def preprocess(self, data, attr):
        cfg = self.cfg

        points = np.array(data['point'][:, 0:3], dtype=np.float32)

        if 'label' not in data.keys() or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data.keys() or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        if cfg.get('t_align', False):
            points_min = np.expand_dims(points.min(0), 0)
            points_min[0, :2] = 0
            points = points - points_min

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

    def init_att_pooling(self, d, dim_output, name):
        att_activation = nn.Linear(d, d)
        setattr(self, name + 'fc', att_activation)

        f_agg = helper_torch.conv2d(True, d, dim_output)
        setattr(self, name + 'mlp', f_agg)

    def init_building_block(self, dim_input, dim_output, name):
        f_pc = helper_torch.conv2d(True, 10, dim_input)
        setattr(self, name + 'mlp1', f_pc)

        self.init_att_pooling(dim_input * 2, dim_output // 2,
                              name + 'att_pooling_1')

        f_xyz = helper_torch.conv2d(True, dim_input, dim_output // 2)
        setattr(self, name + 'mlp2', f_xyz)

        self.init_att_pooling(dim_input * 2, dim_output, name + 'att_pooling_2')

    def init_dilated_res_block(self, dim_input, dim_output, name):
        f_pc = helper_torch.conv2d(True, dim_input, dim_output // 2)
        setattr(self, name + 'mlp1', f_pc)

        self.init_building_block(dim_output // 2, dim_output, name + 'LFA')

        f_pc = helper_torch.conv2d(True,
                                   dim_output,
                                   dim_output * 2,
                                   activation=False)
        setattr(self, name + 'mlp2', f_pc)

        shortcut = helper_torch.conv2d(True,
                                       dim_input,
                                       dim_output * 2,
                                       activation=False)
        setattr(self, name + 'shortcut', shortcut)

    def forward_gather_neighbour(self, pc, neighbor_idx):
        # pc:           BxNxd
        # neighbor_idx: BxNxK
        B, N, K = neighbor_idx.size()
        d = pc.size()[2]

        extended_idx = neighbor_idx.unsqueeze(1).expand(B, d, N, K)
        extended_coords = pc.transpose(-2, -1).unsqueeze(-1).expand(B, d, N, K)
        features = torch.gather(extended_coords, 2, extended_idx)

        return features

    def forward_att_pooling(self, feature_set, name):
        # feature_set: BxdxNxK
        batch_size = feature_set.size()[0]
        num_points = feature_set.size()[2]
        num_neigh = feature_set.size()[3]
        d = feature_set.size()[1]

        f_reshaped = torch.reshape(feature_set.permute(0, 2, 3, 1),
                                   (-1, num_neigh, d))

        m_dense = getattr(self, name + 'fc')
        att_activation = m_dense(f_reshaped)

        m_softmax = nn.Softmax(dim=1)
        att_scores = m_softmax(att_activation)

        # print("att_scores = ", att_scores.shape)
        f_agg = f_reshaped * att_scores
        f_agg = torch.sum(f_agg, dim=1, keepdim=True)
        f_agg = torch.reshape(f_agg, (batch_size, num_points, 1, d))
        f_agg = f_agg.permute(0, 3, 1, 2)

        m_conv2d = getattr(self, name + 'mlp')
        f_agg = m_conv2d(f_agg)

        return f_agg

    def forward_relative_pos_encoding(self, xyz, neigh_idx):
        B, N, K = neigh_idx.size()
        neighbor_xyz = self.forward_gather_neighbour(xyz, neigh_idx)

        xyz_tile = xyz.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        #xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.size()[-1], 1)

        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = torch.sqrt(
            torch.sum(torch.square(relative_xyz), dim=1, keepdim=True))
        relative_feature = torch.cat(
            [relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=1)

        return relative_feature

    def forward_building_block(self, xyz, feature, neigh_idx, name):
        f_xyz = self.forward_relative_pos_encoding(xyz, neigh_idx)
        m_conv2d = getattr(self, name + 'mlp1')
        f_xyz = m_conv2d(f_xyz)

        feature = feature.transpose(1, 2)
        f_neighbours = self.forward_gather_neighbour(
            torch.squeeze(feature, axis=3), neigh_idx)
        f_concat = torch.cat([f_neighbours, f_xyz], axis=1)

        f_pc_agg = self.forward_att_pooling(f_concat, name + 'att_pooling_1')

        m_conv2d = getattr(self, name + 'mlp2')
        f_xyz = m_conv2d(f_xyz)

        f_pc_agg = f_pc_agg.transpose(1, 2)
        f_neighbours = self.forward_gather_neighbour(
            torch.squeeze(f_pc_agg, axis=3), neigh_idx)
        f_concat = torch.cat([f_neighbours, f_xyz], axis=1)
        f_pc_agg = self.forward_att_pooling(f_concat, name + 'att_pooling_2')

        return f_pc_agg

    def forward_dilated_res_block(self, feature, xyz, neigh_idx, dim_output,
                                  name):
        m_conv2d = getattr(self, name + 'mlp1')
        f_pc = m_conv2d(feature)

        f_pc = self.forward_building_block(xyz, f_pc, neigh_idx, name + 'LFA')

        m_conv2d = getattr(self, name + 'mlp2')
        f_pc = m_conv2d(f_pc)

        m_conv2d = getattr(self, name + 'shortcut')
        shortcut = m_conv2d(feature)

        m_leakyrelu = nn.LeakyReLU(0.2)

        result = m_leakyrelu(f_pc + shortcut)
        return result

    def forward(self, inputs):
        device = self.device
        xyz = [arr.to(device) for arr in inputs['xyz']]
        neigh_idx = [arr.to(device) for arr in inputs['neigh_idx']]
        interp_idx = [arr.to(device) for arr in inputs['interp_idx']]
        sub_idx = [arr.to(device) for arr in inputs['sub_idx']]
        feature = inputs['features'].to(device)

        m_dense = getattr(self, 'fc0')
        feature = m_dense(feature).transpose(-2, -1).unsqueeze(-1)

        m_bn = getattr(self, 'batch_normalization')
        feature = m_bn(feature)

        m_leakyrelu = nn.LeakyReLU(0.2)
        feature = m_leakyrelu(feature)

        # B d N 1
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
        feature = m_conv2d(f_encoder_list[-1])

        # Decoder
        f_decoder_list = []
        for j in range(self.cfg.num_layers):
            f_interp_i = self.nearest_interpolation(feature, interp_idx[-j - 1])
            name = 'Decoder_layer_' + str(j)

            m_transposeconv2d = getattr(self, name)
            concat_feature = torch.cat([f_encoder_list[-j - 2], f_interp_i],
                                       dim=1)
            f_decoder_i = m_transposeconv2d(concat_feature)

            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)

        m_conv2d = getattr(self, 'fc1')
        f_layer_fc1 = m_conv2d(f_decoder_list[-1])

        m_conv2d = getattr(self, 'fc2')
        f_layer_fc2 = m_conv2d(f_layer_fc1)

        f_layer_drop = self.m_dropout(f_layer_fc2)

        test_hidden = f_layer_fc2.permute(0, 2, 3, 1)

        m_conv2d = getattr(self, 'fc')
        f_layer_fc3 = m_conv2d(f_layer_drop)

        f_out = f_layer_fc3.squeeze(3).transpose(1, 2)

        return f_out

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        Args:
            feature: [B, d, N, 1] input features matrix
            pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling

        Returns:
             pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(3)
        num_neigh = pool_idx.size()[2]
        batch_size = feature.size()[0]
        d = feature.size()[1]

        pool_idx = torch.reshape(pool_idx, (batch_size, -1))

        pool_idx = pool_idx.unsqueeze(2).expand(batch_size, -1, d)

        feature = feature.transpose(1, 2)
        pool_features = torch.gather(feature, 1, pool_idx)
        pool_features = torch.reshape(pool_features,
                                      (batch_size, -1, num_neigh, d))
        pool_features, _ = torch.max(pool_features, 2, keepdim=True)
        pool_features = pool_features.permute(0, 3, 1, 2)

        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        Args:
            feature: [B, d, N] input features matrix
            interp_idx: [B, up_num_points, 1] nearest neighbour index

        Returns:
             [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(3)
        d = feature.size(1)
        batch_size = interp_idx.size()[0]
        up_num_points = interp_idx.size()[1]

        interp_idx = torch.reshape(interp_idx, (batch_size, up_num_points))
        interp_idx = interp_idx.unsqueeze(1).expand(batch_size, d, -1)

        interpolatedim_features = torch.gather(feature, 2, interp_idx)
        interpolatedim_features = interpolatedim_features.unsqueeze(3)
        return interpolatedim_features


MODEL._register_module(RandLANet, 'torch')
