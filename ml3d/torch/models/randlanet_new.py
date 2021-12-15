import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import KDTree

from .base_model import BaseModel
from ..utils import helper_torch
from ..dataloaders import DefaultBatcher
from ..modules.losses import filter_valid_label
from ...datasets.utils import (DataProcessing, trans_normalize, trans_augment,
                               trans_crop_pc)
from ...utils import MODEL


class RandLANet(BaseModel):
    """Class defining RandLANet, a Semantic Segmentation model.
    Based on the architecture
    https://arxiv.org/abs/1911.11236#
    """

    def __init__(
            self,
            name='RandLANet',
            num_neighbors=16,
            num_layers=4,
            num_points=4096 * 11,
            num_classes=19,
            ignored_label_inds=[0],
            sub_sampling_ratio=[4, 4, 4, 4],
            in_channels=3,  # 3 + feature_dimension.
            dim_features=8,
            dim_output=[16, 64, 128, 256],
            grid_size=0.06,
            batcher='DefaultBatcher',
            ckpt_path=None,
            weight_decay=0.0,
            **kwargs):

        super().__init__(name=name,
                         num_neighbors=num_neighbors,
                         num_layers=num_layers,
                         num_points=num_points,
                         num_classes=num_classes,
                         ignored_label_inds=ignored_label_inds,
                         sub_sampling_ratio=sub_sampling_ratio,
                         in_channels=in_channels,
                         dim_features=dim_features,
                         dim_output=dim_output,
                         grid_size=grid_size,
                         batcher=batcher,
                         ckpt_path=ckpt_path,
                         weight_decay=weight_decay,
                         **kwargs)
        cfg = self.cfg

        self.fc0 = nn.Linear(cfg.in_channels, cfg.dim_features)
        self.bn0 = nn.BatchNorm2d(cfg.dim_features, eps=1e-6, momentum=0.01)

        # Encoder
        self.encoder = []
        encoder_dim_list = []
        dim_feature = cfg.dim_features
        for i in range(cfg.num_layers):
            self.encoder.append(
                LocalFeatureAggregation(dim_feature, cfg.dim_output[i],
                                        cfg.num_neighbors))
            dim_feature = 2 * cfg.dim_output[i]
            if i == 0:
                encoder_dim_list.append(dim_feature)
            encoder_dim_list.append(dim_feature)

        self.encoder = nn.ModuleList(self.encoder)

        self.mlp = SharedMLP(dim_feature, dim_feature, activation_fn=nn.ReLU())

        # Decoder
        self.decoder = []
        for i in range(cfg.num_layers):
            self.decoder.append(
                SharedMLP(encoder_dim_list[-i - 2] + dim_feature,
                          encoder_dim_list[-i - 2],
                          transpose=True,
                          bn=True,
                          activation_fn=nn.ReLU()))
            dim_feature = encoder_dim_list[-i - 2]

        self.decoder = nn.ModuleList(self.decoder)

        self.fc1 = nn.Sequential(
            SharedMLP(dim_feature, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(0.5), SharedMLP(32, cfg.num_classes))

    def preprocess(self, data, attr):
        cfg = self.cfg

        points = np.array(data['point'][:, 0:3], dtype=np.float32)

        if 'label' not in data or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data or data['feat'] is None:
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

        if feat is not None:
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

        assert cfg.in_channels == feat.shape[
            1], "Wrong feature dimension, please update dim_input(3 + feature_dimension) in config"

        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            # TODO: Replace with Open3D KNN
            neighbour_idx = DataProcessing.knn_search(pc, pc, cfg.num_neighbors)

            sub_points = pc[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            up_i = DataProcessing.knn_search(sub_points, pc, 1)
            input_points.append(pc)
            input_neighbors.append(neighbour_idx.astype(np.int64))
            input_pools.append(pool_i.astype(np.int64))
            input_up_samples.append(up_i.astype(np.int64))
            pc = sub_points

        inputs['coords'] = input_points
        inputs['neighbor_indices'] = input_neighbors
        inputs['sub_idx'] = input_pools
        inputs['interp_idx'] = input_up_samples
        inputs['features'] = feat
        inputs['point_inds'] = selected_idxs
        inputs['labels'] = label.astype(np.int64)

        return inputs

    def forward(self, inputs):
        """Forward pass for RandLANet

        Args:
            inputs: torch.Tensor, shape (B, N, d_in)
                input points

        Returns
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point

        """
        cfg = self.cfg
        feat = inputs['features'].to(self.device)  # (B, N, in_channels)
        coords_list = [arr.to(self.device) for arr in inputs['coords']]
        neighbor_indices_list = [
            arr.to(self.device) for arr in inputs['neighbor_indices']
        ]
        subsample_indices_list = [
            arr.to(self.device) for arr in inputs['sub_idx']
        ]
        interpolation_indices_list = [
            arr.to(self.device) for arr in inputs['interp_idx']
        ]

        feat = self.fc0(feat).transpose(-2, -1).unsqueeze(
            -1)  # (B, dim_feature, N, 1)
        feat = self.bn0(feat)  # (B, d, N, 1)

        # Encoder
        encoder_feat_list = []
        for i in range(cfg.num_layers):
            feat_encoder_i = self.encoder[i](coords_list[i], feat,
                                             neighbor_indices_list[i])
            feat_sampled_i = self.random_sample(feat_encoder_i,
                                                subsample_indices_list[i])
            if i == 0:
                encoder_feat_list.append(feat_encoder_i.clone())
            encoder_feat_list.append(feat_sampled_i.clone())
            feat = feat_sampled_i

        feat = self.mlp(feat)

        # Decoder
        for i in range(cfg.num_layers):
            feat_interpolation_i = self.nearest_interpolation(
                feat, interpolation_indices_list[-i - 1])
            feat_decoder_i = torch.cat(
                [encoder_feat_list[-i - 2], feat_interpolation_i], dim=1)
            feat_decoder_i = self.decoder[i](feat_decoder_i)
            feat = feat_decoder_i

        scores = self.fc1(feat)

        return scores.squeeze(3).transpose(1, 2)

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

    def inference_begin():
        pass

    def inference_preprocess():
        pass

    def inference_end():
        pass


MODEL._register_module(RandLANet, 'torch')


class SharedMLP(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 transpose=False,
                 bn=False,
                 activation_fn=None):
        super(SharedMLP, self).__init__()

        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
            )
        else:
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size,
                                  stride=stride)

        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6,
                                         momentum=0.01) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        """Forward pass of the Module.

        Args:
            input: torch.Tensor of shape (B, dim_in, N, K)

        Returns:
            torch.Tensor, shape (B, dim_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):

    def __init__(self, d, num_neighbors):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        # self.device = device

    def gather_neighbor(self, coords, neighbor_indices):
        """Gather features based on neighbor indices.

        Args:
            coords: torch.Tensor of shape (B, N, d)
            neighbor_indices: torch.Tensor of shape (B, N, K)
        
        Returns:
            gathered neighbors of shape (B, dim, N, K)

        """
        B, N, K = neighbor_indices.size()
        dim = coords.shape[2]

        extended_indices = neighbor_indices.unsqueeze(1).expand(B, dim, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(
            B, dim, N, K)
        neighbor_coords = torch.gather(extended_coords, 2,
                                       extended_indices)  # (B, dim, N, K)

        return neighbor_coords

    def forward(self, coords, features, neighbor_indices):
        """Forward pass of the Module.

        Args:
            coords: coordinates of the pointcloud
                torch.Tensor of shape (B, N, 3)
            features: features of the pointcloud.
                torch.Tensor of shape (B, d, N, 1)
            neighbor_indices: indices of k neighbours.
                torch.Tensor of shape (B, N, K)

        Returns:
            torch.Tensor of shape (B, 2*d, N, K)
        """
        # finding neighboring points
        B, N, K = neighbor_indices.size()

        neighbor_coords = self.gather_neighbor(coords, neighbor_indices)

        extended_coords = coords.transpose(-2,
                                           -1).unsqueeze(-1).expand(B, 3, N, K)

        relative_pos = extended_coords - neighbor_coords
        relative_dist = torch.sqrt(
            torch.sum(torch.square(relative_pos), dim=1, keepdim=True))

        relative_features = torch.cat(
            [relative_dist, relative_pos, extended_coords, neighbor_coords],
            dim=1)
        relative_features = self.mlp(relative_features)

        neighbor_features = self.gather_neighbor(
            features.transpose(1, 2).squeeze(3), neighbor_indices)

        return torch.cat([neighbor_features, relative_features], dim=1)


class AttentivePooling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False), nn.Softmax(dim=-2))
        self.mlp = SharedMLP(in_channels,
                             out_channels,
                             bn=True,
                             activation_fn=nn.ReLU())

    def forward(self, x):
        """Forward pass of the Module.

        Args:
            x: torch.Tensor of shape (B, dim_in, N, K).

        Returns:
            torch.Tensor of shape (B, d_out, N, 1).
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1,
                             keepdim=True)  # shape (B, d_in, N, 1)

        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):

    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2 * d_out)
        self.shortcut = SharedMLP(d_in, 2 * d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out // 2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(d_out // 2, num_neighbors)

        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, feat, neighbor_indices):
        """Forward pass of the Module.

        Args:
            coords: coordinates of the pointcloud
                torch.Tensor of shape (B, N, 3).
            feat: features of the pointcloud.
                torch.Tensor of shape (B, d, N, 1)
            neighbor_indices: Indices of neighbors.

        Returns:
            torch.Tensor of shape (B, 2*d_out, N, 1).

        """
        # knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)

        x = self.mlp1(feat)

        x = self.lse1(coords, x, neighbor_indices)
        x = self.pool1(x)

        x = self.lse2(coords, x, neighbor_indices)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(feat))
