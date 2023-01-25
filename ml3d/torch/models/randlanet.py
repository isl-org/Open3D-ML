import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import KDTree

from .base_model import BaseModel
from ..dataloaders import DefaultBatcher
from ...datasets.augment import SemsegAugmentation
from ..modules.losses import filter_valid_label
from ...datasets.utils import DataProcessing
from ...utils import MODEL


class RandLANet(BaseModel):
    """Class defining RandLANet, a Semantic Segmentation model.  Based on the
    architecture from the paper `RandLA-Net: Efficient Semantic Segmentation of
    Large-Scale Point Clouds <https://arxiv.org/abs/1911.11236>`__.

    RandLA-Net is an efficient and lightweight neural architecture which
    directly infer per-point semantics for large-scale point clouds. The key
    approach is to use random point sampling instead of more complex point
    selection approaches.  Although remarkably computation and memory
    efficient, random sampling can discard key features by chance. To overcome
    this, we introduce a novel local feature aggregation module to
    progressively increase the receptive field for each 3D point, thereby
    effectively preserving geometric details.

    **Architecture**

    .. image:: https://user-images.githubusercontent.com/23613902/150006228-34fb9e04-76b6-4022-af08-c308da6dcaae.png
        :width: 100%

    References:
        https://github.com/QingyongHu/RandLA-Net
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
            augment={},
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
                         augment=augment,
                         **kwargs)
        cfg = self.cfg
        self.augmenter = SemsegAugmentation(cfg.augment, seed=self.rng)

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

        self.mlp = SharedMLP(dim_feature,
                             dim_feature,
                             activation_fn=nn.LeakyReLU(0.2))

        # Decoder
        self.decoder = []
        for i in range(cfg.num_layers):
            self.decoder.append(
                SharedMLP(encoder_dim_list[-i - 2] + dim_feature,
                          encoder_dim_list[-i - 2],
                          transpose=True,
                          activation_fn=nn.LeakyReLU(0.2)))
            dim_feature = encoder_dim_list[-i - 2]

        self.decoder = nn.ModuleList(self.decoder)

        self.fc1 = nn.Sequential(
            SharedMLP(dim_feature, 64, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(64, 32, activation_fn=nn.LeakyReLU(0.2)), nn.Dropout(0.5),
            SharedMLP(32, cfg.num_classes, bn=False))

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

        split = attr['split']
        data = dict()

        if feat is None:
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
        # If num_workers > 0, use new RNG with unique seed for each thread.
        # Else, use default RNG.
        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(
                torch.utils.data.get_worker_info().seed +
                torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng

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

        augment_cfg = self.cfg.get('augment', {}).copy()
        val_augment_cfg = {}
        if 'recenter' in augment_cfg:
            val_augment_cfg['recenter'] = augment_cfg.pop('recenter')
        if 'normalize' in augment_cfg:
            val_augment_cfg['normalize'] = augment_cfg.pop('normalize')

        self.augmenter.augment(pc, feat, label, val_augment_cfg, seed=rng)

        if attr['split'] in ['training', 'train']:
            pc, feat, label = self.augmenter.augment(pc,
                                                     feat,
                                                     label,
                                                     augment_cfg,
                                                     seed=rng)

        if feat is None:
            feat = pc.copy()
        else:
            feat = np.concatenate([pc, feat], axis=1)

        if cfg.in_channels != feat.shape[1]:
            raise RuntimeError(
                "Wrong feature dimension, please update in_channels(3 + feature_dimension) in config"
            )

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

        l_relu = nn.LeakyReLU(0.2)
        feat = l_relu(feat)

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
                                     **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler

    def get_loss(self, Loss, results, inputs, device):
        """Calculate the loss on output of the model.

        Args:
            Loss: Object of type `SemSegLoss`.
            results: Output of the model (B, N, C).
            inputs: Input of the model.
            device: device(cpu or cuda).

        Returns:
            Returns loss, labels and scores.

        """
        cfg = self.cfg
        labels = inputs['data']['labels']

        scores, labels = filter_valid_label(results, labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def inference_begin(self, data):
        self.test_smooth = 0.95
        attr = {'split': 'test'}
        self.inference_ori_data = data
        self.inference_data = self.preprocess(data, attr)
        self.inference_proj_inds = self.inference_data['proj_inds']
        num_points = self.inference_data['search_tree'].data.shape[0]
        self.possibility = self.rng.random(num_points) * 1e-3
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

    def update_probs(self, inputs, results, test_probs):
        """Update test probabilities with probs from current tested patch.

        Args:
            inputs: input to the model.
            results: output of the model.
            test_probs: probabilities for whole pointcloud

        Returns:
            updated probabilities

        """
        self.test_smooth = 0.95

        for b in range(results.size()[0]):

            result = torch.reshape(results[b], (-1, self.cfg.num_classes))
            probs = torch.nn.functional.softmax(result, dim=-1)
            probs = probs.cpu().data.numpy()
            inds = inputs['data']['point_inds'][b]

            test_probs[inds] = self.test_smooth * test_probs[inds] + (
                1 - self.test_smooth) * probs

        return test_probs


MODEL._register_module(RandLANet, 'torch')


class SharedMLP(nn.Module):
    """Module consisting of commonly used layers conv, batchnorm
    and any activation function.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 transpose=False,
                 bn=True,
                 activation_fn=None):
        super(SharedMLP, self).__init__()

        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=(kernel_size - 1) // 2)
        else:
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=(kernel_size - 1) // 2)

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
    """This module computes k neighbour feature encoding for each point.
    Encoding consists of absolute distance, relative distance, positions.
    """

    def __init__(self, dim_in, dim_out, num_neighbors, encode_pos=False):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(dim_in, dim_out, activation_fn=nn.LeakyReLU(0.2))
        self.encode_pos = encode_pos

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

    def forward(self,
                coords,
                features,
                neighbor_indices,
                relative_features=None):
        """Forward pass of the Module.

        Args:
            coords: coordinates of the pointcloud
                torch.Tensor of shape (B, N, 3)
            features: features of the pointcloud.
                torch.Tensor of shape (B, d, N, 1)
            neighbor_indices: indices of k neighbours.
                torch.Tensor of shape (B, N, K)
            relative_features: relative neighbor features calculated
              on first pass. Required for second pass.

        Returns:
            torch.Tensor of shape (B, 2*d, N, K)

        """
        # finding neighboring points
        B, N, K = neighbor_indices.size()

        if self.encode_pos:
            neighbor_coords = self.gather_neighbor(coords, neighbor_indices)

            extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(
                B, 3, N, K)

            relative_pos = extended_coords - neighbor_coords
            relative_dist = torch.sqrt(
                torch.sum(torch.square(relative_pos), dim=1, keepdim=True))

            relative_features = torch.cat(
                [relative_dist, relative_pos, extended_coords, neighbor_coords],
                dim=1)

        else:
            if relative_features is None:
                raise ValueError(
                    "LocalSpatialEncoding: Require relative_features for second pass."
                )

        relative_features = self.mlp(relative_features)

        neighbor_features = self.gather_neighbor(
            features.transpose(1, 2).squeeze(3), neighbor_indices)

        return torch.cat([neighbor_features, relative_features],
                         dim=1), relative_features


class AttentivePooling(nn.Module):
    """This module pools down k neighbour features to a single encoding
    using weighted average with attention scores.
    """

    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(nn.Linear(in_channels, in_channels),
                                      nn.Softmax(dim=-2))
        self.mlp = SharedMLP(in_channels,
                             out_channels,
                             activation_fn=nn.LeakyReLU(0.2))

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
    """The neighbour features returned from LocalSpatialEncoding
    and pooled from AttentivePooling are aggregated and processed
    in multiple layers in this module.
    """

    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.lse1 = LocalSpatialEncoding(10,
                                         d_out // 2,
                                         num_neighbors,
                                         encode_pos=True)
        self.pool1 = AttentivePooling(d_out, d_out // 2)

        self.lse2 = LocalSpatialEncoding(d_out // 2, d_out // 2, num_neighbors)
        self.pool2 = AttentivePooling(d_out, d_out)
        self.mlp2 = SharedMLP(d_out, 2 * d_out)

        self.shortcut = SharedMLP(d_in, 2 * d_out)
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
        x = self.mlp1(feat)

        x, neighbor_features = self.lse1(coords, x, neighbor_indices)
        x = self.pool1(x)

        x, _ = self.lse2(coords,
                         x,
                         neighbor_indices,
                         relative_features=neighbor_features)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(feat))
