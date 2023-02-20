import numpy as np
import torch
import torch.nn as nn
import torch.utils.dlpack
import open3d.core as o3c

from sklearn.neighbors import KDTree
from open3d.ml.torch.ops import knn_search

from .base_model import BaseModel
from ...utils import MODEL
from ..modules.losses import filter_valid_label
from ...datasets.augment import SemsegAugmentation
from ...datasets.utils import DataProcessing
from ..utils.pointnet.pointnet2_utils import furthest_point_sample_v2


class PointTransformer(BaseModel):
    """Semantic Segmentation model. Based on PointTransformer architecture
    https://arxiv.org/pdf/2012.09164.pdf

    Uses Encoder-Decoder architecture with Transformer layers.

    Attributes:
        name: Name of model.
          Default to "PointTransformer".
        blocks: Number of Bottleneck layers.
        in_channels: Number of features(default 6).
        num_classes: Number of classes.
        voxel_size: Voxel length for subsampling.
        max_voxels: Maximum number of voxels.
        batcher: Batching method for dataloader.
        augment: dictionary for augmentation.
    """

    def __init__(self,
                 name="PointTransformer",
                 blocks=[2, 2, 2, 2, 2],
                 in_channels=6,
                 num_classes=13,
                 voxel_size=0.04,
                 max_voxels=80000,
                 batcher='ConcatBatcher',
                 augment=None,
                 **kwargs):
        super(PointTransformer, self).__init__(name=name,
                                               blocks=blocks,
                                               in_channels=in_channels,
                                               num_classes=num_classes,
                                               voxel_size=voxel_size,
                                               max_voxels=max_voxels,
                                               batcher=batcher,
                                               augment=augment,
                                               **kwargs)
        cfg = self.cfg
        self.in_channels = in_channels
        self.augmenter = SemsegAugmentation(cfg.augment)
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        block = Bottleneck

        self.encoders = nn.ModuleList()
        for i in range(5):
            self.encoders.append(
                self._make_enc(
                    block,
                    planes[i],
                    blocks[i],
                    share_planes,
                    stride=stride[i],
                    nsample=nsample[i]))  # N/1, N/4, N/16, N/64, N/256

        self.decoders = nn.ModuleList()
        for i in range(4, -1, -1):
            self.decoders.append(
                self._make_dec(block,
                               planes[i],
                               2,
                               share_planes,
                               nsample=nsample[i],
                               is_head=True if i == 4 else False))

        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]),
                                 nn.BatchNorm1d(planes[0]),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(planes[0], num_classes))

    def _make_enc(self,
                  block,
                  planes,
                  blocks,
                  share_planes=8,
                  stride=1,
                  nsample=16):
        """Private method to create encoder.

        Args:
            block: Bottleneck block consisting transformer layers.
            planes: list of feature dimension.
            blocks: Number of `block` layers.
            share_planes: Number of common planes for transformer.
            stride: stride for pooling.
            nsample: number of neighbour to sample.

        Returns:
            Returns encoder object.
        """
        layers = []
        layers.append(
            TransitionDown(self.in_planes, planes * block.expansion, stride,
                           nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes,
                      self.in_planes,
                      share_planes,
                      nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self,
                  block,
                  planes,
                  blocks,
                  share_planes=8,
                  nsample=16,
                  is_head=False):
        """Private method to create decoder.

        Args:
            block: Bottleneck block consisting transformer layers.
            planes: list of feature dimension.
            blocks: Number of `block` layers.
            share_planes: Number of common planes for transformer.
            nsample: number of neighbour to sample.
            is_head: bool type for head layer.

        Returns:
            Returns decoder object.
        """
        layers = []
        layers.append(
            TransitionUp(self.in_planes,
                         None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes,
                      self.in_planes,
                      share_planes,
                      nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, batch):
        """Forward pass for the model.

        Args:
            inputs: A dict object for inputs with following keys
                point (tf.float32): Input pointcloud (N,3)
                feat (tf.float32): Input features (N, 3)
                row_splits (tf.int64): row splits for batches (b+1,)

        Returns:
            Returns the probability distribution.
        """
        points = [batch.point]  # (n, 3)
        feats = [batch.feat]  # (n, c)
        row_splits = [batch.row_splits]  # (b)

        feats[0] = points[0] if self.in_channels == 3 else torch.cat(
            (points[0], feats[0]), 1)

        for i in range(5):
            p, f, r = self.encoders[i]([points[i], feats[i], row_splits[i]])
            points.append(p)
            feats.append(f)
            row_splits.append(r)

        for i in range(4, -1, -1):
            if i == 4:
                feats[i + 1] = self.decoders[4 - i][1:]([
                    points[i + 1], self.decoders[4 - i][0](
                        [points[i + 1], feats[i + 1], row_splits[i + 1]]),
                    row_splits[i + 1]
                ])[1]
            else:
                feats[i + 1] = self.decoders[4 - i][1:]([
                    points[i + 1], self.decoders[4 - i][0](
                        [points[i + 1], feats[i + 1], row_splits[i + 1]],
                        [points[i + 2], feats[i + 2], row_splits[i + 2]]),
                    row_splits[i + 1]
                ])[1]

        feat = self.cls(feats[1])
        return feat

    def preprocess(self, data, attr):
        """Data preprocessing function.

        This function is called before training to preprocess the data from a
        dataset. It consists of subsampling pointcloud with voxelization.

        Args:
            data: A sample from the dataset.
            attr: The corresponding attributes.

        Returns:
            Returns the preprocessed data

        """
        cfg = self.cfg
        points = np.array(data['point'], dtype=np.float32)

        if data.get('label') is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if data.get('feat') is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        data = dict()

        if (cfg.voxel_size):
            points_min = np.min(points, 0)
            points -= points_min

            if (feat is None):
                sub_points, sub_labels = DataProcessing.grid_subsampling(
                    points, labels=labels, grid_size=cfg.voxel_size)
                sub_feat = None
            else:
                sub_points, sub_feat, sub_labels = DataProcessing.grid_subsampling(
                    points,
                    features=feat,
                    labels=labels,
                    grid_size=cfg.voxel_size)
        else:
            sub_points, sub_feat, sub_labels = points, feat, labels

        search_tree = KDTree(sub_points)

        data['point'] = sub_points
        data['feat'] = sub_feat
        data['label'] = sub_labels
        data['search_tree'] = search_tree

        if attr['split'] in ["test", "testing"]:
            proj_inds = np.squeeze(
                search_tree.query(points, return_distance=False))
            proj_inds = proj_inds.astype(np.int32)
            data['proj_inds'] = proj_inds

        return data

    def transform(self, data, attr):
        """Transform function for the point cloud and features.

        This function is called after preprocess method. It consists
        of calling augmentation and normalizing the pointcloud.

        Args:
            data: A sample from the dataset.
            attr: The corresponding attributes.

        Returns:
            Returns dictionary data with keys
            (point, feat, label).

        """
        cfg = self.cfg
        points = data['point']
        feat = data['feat']
        labels = data['label']

        if attr['split'] in ['training', 'train']:
            points, feat, labels = self.augmenter.augment(
                points, feat, labels, self.cfg.get('augment', None))

        if attr['split'] not in ['test', 'testing']:
            if cfg.max_voxels and data['label'].shape[0] > cfg.max_voxels:
                init_idx = np.random.randint(
                    labels.shape[0]
                ) if 'train' in attr['split'] else labels.shape[0] // 2
                crop_idx = np.argsort(
                    np.sum(np.square(points - points[init_idx]),
                           1))[:cfg.max_voxels]
                if feat is not None:
                    points, feat, labels = points[crop_idx], feat[
                        crop_idx], labels[crop_idx]
                else:
                    points, labels = points[crop_idx], labels[crop_idx]

        points_min, points_max = np.min(points, 0), np.max(points, 0)
        points -= (points_min + points_max) / 2.0

        data['point'] = torch.from_numpy(points).to(torch.float32)
        if feat is not None:
            data['feat'] = torch.from_numpy(feat).to(torch.float32) / 255.0
        data['label'] = torch.from_numpy(labels).to(torch.int64)

        return data

    def update_probs(self, inputs, results, test_probs):
        result = results.reshape(-1, self.cfg.num_classes)
        probs = torch.nn.functional.softmax(result, dim=-1).cpu().data.numpy()

        self.trans_point_sampler(patchwise=False)

        return probs

    def inference_begin(self):
        data = self.preprocess(data, {'split': 'test'})
        data = self.transform(data, {'split': 'test'})

        self.inference_input = data

    def inference_preprocess(self):
        return self.inference_input

    def inference_end(self, inputs, results):
        results = torch.reshape(results, (-1, self.cfg.num_classes))

        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()

        probs = np.reshape(results, [-1, self.cfg.num_classes])
        reproj_inds = self.inference_input['proj_inds']
        probs = probs[reproj_inds]

        pred_l = np.argmax(probs, 1)

        return {'predict_labels': pred_l, 'predict_scores': probs}

    def get_loss(self, sem_seg_loss, results, inputs, device):
        """Calculate the loss on output of the model.

        Args:
            sem_seg_loss: Object of type `SemSegLoss`.
            results: Output of the model.
            inputs: Input of the model.
            device: device(cpu or cuda).

        Returns:
            Returns loss, labels and scores.
        """
        cfg = self.cfg
        labels = inputs['data'].label

        scores, labels = filter_valid_label(results, labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        loss = sem_seg_loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.SGD(self.parameters(), **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(cfg_pipeline.max_epoch * 0.6),
                int(cfg_pipeline.max_epoch * 0.8)
            ],
            gamma=0.1)

        return optimizer, scheduler


MODEL._register_module(PointTransformer, 'torch')


class Transformer(nn.Module):
    """Transformer layer of the model, uses self attention."""

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        """Constructor for Transformer Layer.

        Args:
            in_planes (int): Number of input planes.
            out_planes (int): Number of output planes.
            share_planes (int): Number of shared planes.
            nsample (int): Number of neighbours.

        """
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes // share_planes),
            nn.BatchNorm1d(mid_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes,
                      out_planes // share_planes),  # Verify
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        """Forward call for Transformer.

        Args:
            pxo: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)

        Returns:
            Transformer features.

        """
        point, feat, row_splits = pxo  # (n, 3), (n, c), (b)
        feat_q, feat_k, feat_v = self.linear_q(feat), self.linear_k(
            feat), self.linear_v(feat)  # (n, c)
        feat_k = queryandgroup(self.nsample,
                               point,
                               point,
                               feat_k,
                               None,
                               row_splits,
                               row_splits,
                               use_xyz=True)  # (n, nsample, 3+c)
        feat_v = queryandgroup(self.nsample,
                               point,
                               point,
                               feat_v,
                               None,
                               row_splits,
                               row_splits,
                               use_xyz=False)  # (n, nsample, c)
        point_r, feat_k = feat_k[:, :, 0:3], feat_k[:, :, 3:]

        for i, layer in enumerate(self.linear_p):
            point_r = layer(point_r.transpose(1, 2).contiguous()).transpose(
                1, 2).contiguous() if i == 1 else layer(
                    point_r)  # (n, nsample, c)

        w = feat_k - feat_q.unsqueeze(1) + point_r.view(
            point_r.shape[0], point_r.shape[1], self.out_planes //
            self.mid_planes, self.mid_planes).sum(2)  # (n, nsample, c)

        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(
                1, 2).contiguous() if i % 3 == 0 else layer(w)

        w = self.softmax(w)  # (n, nsample, c)
        n, nsample, c = feat_v.shape
        s = self.share_planes
        feat = ((feat_v + point_r).view(n, nsample, s, c // s) *
                w.unsqueeze(2)).sum(1).view(n, c)

        return feat


class TransitionDown(nn.Module):
    """TransitionDown layer for PointTransformer.

    Subsamples points and increase receptive field.
    """

    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        """Constructor for TransitionDown Layer.

        Args:
            in_planes (int): Number of input planes.
            out_planes (int): Number of output planes.
            stride (int): subsampling factor.
            nsample (int): Number of neighbours.

        """
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        """Forward call for TransitionDown

        Args:
            pxo: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)

        Returns:
            List of point, feat, row_splits.

        """
        point, feat, row_splits = pxo  # (n, 3), (n, c), (b+1)
        if self.stride != 1:
            new_row_splits = [0]
            count = 0
            for i in range(1, row_splits.shape[0]):
                count += (row_splits[i].item() -
                          row_splits[i - 1].item()) // self.stride
                new_row_splits.append(count)

            new_row_splits = torch.LongTensor(new_row_splits).to(
                row_splits.device)
            idx = furthest_point_sample_v2(point, row_splits,
                                           new_row_splits)  # (m)
            new_point = point[idx.long(), :]  # (m, 3)
            feat = queryandgroup(self.nsample,
                                 point,
                                 new_point,
                                 feat,
                                 None,
                                 row_splits,
                                 new_row_splits,
                                 use_xyz=True)  # (m, nsample, 3+c)
            feat = self.relu(
                self.bn(self.linear(feat).transpose(
                    1, 2).contiguous()))  # (m, c, nsample)
            feat = self.pool(feat).squeeze(-1)  # (m, c)
            point, row_splits = new_point, new_row_splits
        else:
            feat = self.relu(self.bn(self.linear(feat)))  # (n, c)
        return [point, feat, row_splits]


class TransitionUp(nn.Module):
    """Decoder layer for PointTransformer.

    Interpolate points based on corresponding encoder layer.
    """

    def __init__(self, in_planes, out_planes=None):
        """Constructor for TransitionUp Layer.

        Args:
            in_planes (int): Number of input planes.
            out_planes (int): Number of output planes.

        """
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes),
                                         nn.BatchNorm1d(in_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes),
                                         nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes),
                                         nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes),
                                         nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        """Forward call for TransitionUp

        Args:
            pxo1: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)
            pxo2: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)

        Returns:
            Interpolated features.

        """
        if pxo2 is None:
            _, feat, row_splits = pxo1  # (n, 3), (n, c), (b)
            feat_tmp = []
            for i in range(0, row_splits.shape[0] - 1):
                start_i, end_i, count = row_splits[i], row_splits[
                    i + 1], row_splits[i + 1] - row_splits[i]
                feat_b = feat[start_i:end_i, :]
                feat_b = torch.cat(
                    (feat_b, self.linear2(feat_b.sum(0, True) / count).repeat(
                        count, 1)), 1)
                feat_tmp.append(feat_b)
            feat = torch.cat(feat_tmp, 0)
            feat = self.linear1(feat)
        else:
            point_1, feat_1, row_splits_1 = pxo1
            point_2, feat_2, row_splits_2 = pxo2
            feat = self.linear1(feat_1) + interpolation(
                point_2, point_1, self.linear2(feat_2), row_splits_2,
                row_splits_1)
        return feat


class Bottleneck(nn.Module):
    """Bottleneck layer for PointTransformer.

    Block of layers using Transformer layer as building block.
    """
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        """Constructor for Bottleneck Layer.

        Args:
            in_planes (int): Number of input planes.
            planes (int): Number of output planes.
            share_planes (int): Number of shared planes.
            nsample (int): Number of neighbours.

        """
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = Transformer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        """Forward call for Bottleneck

        Args:
            pxo: [point, feat, row_splits] with shapes
                (n, 3), (n, c) and (b+1,)

        Returns:
            List of point, feat, row_splits.

        """
        point, feat, row_splits = pxo  # (n, 3), (n, c), (b)
        identity = feat
        feat = self.relu(self.bn1(self.linear1(feat)))
        feat = self.relu(self.bn2(self.transformer2([point, feat, row_splits])))
        feat = self.bn3(self.linear3(feat))
        feat += identity
        feat = self.relu(feat)
        return [point, feat, row_splits]


def queryandgroup(nsample,
                  points,
                  queries,
                  feat,
                  idx,
                  points_row_splits,
                  queries_row_splits,
                  use_xyz=True):
    """Find nearest neighbours and returns grouped features.

    Args:
        nsample: Number of neighbours (k).
        points: Input pointcloud (n, 3).
        queries: Queries for Knn (m, 3).
        feat: features (n, c).
        idx: Optional knn index list.
        points_row_splits: row_splits for batching points.
        queries_row_splits: row_splits for batching queries.
        use_xyz: Whether to return xyz concatenated with features.

    Returns:
        Returns grouped features (m, nsample, c) or (m, nsample, 3+c).

    """
    if not (points.is_contiguous and queries.is_contiguous() and
            feat.is_contiguous()):
        raise ValueError("queryandgroup (points/queries/feat not contiguous)")
    if queries is None:
        queries = points
    if idx is None:
        idx = knn_batch(points,
                        queries,
                        k=nsample,
                        points_row_splits=points_row_splits,
                        queries_row_splits=queries_row_splits,
                        return_distances=False)

    n, m, c = points.shape[0], queries.shape[0], feat.shape[1]
    grouped_xyz = points[idx.view(-1).long(), :].view(m, nsample,
                                                      3)  # (m, nsample, 3)
    grouped_xyz -= queries.unsqueeze(1)  # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample,
                                                     c)  # (m, nsample, c)

    if use_xyz:
        return torch.cat((grouped_xyz, grouped_feat), -1)  # (m, nsample, 3+c)
    else:
        return grouped_feat


def knn_batch(points,
              queries,
              k,
              points_row_splits,
              queries_row_splits,
              return_distances=True):
    """K nearest neighbour with batch support.

    Args:
        points: Input pointcloud.
        queries: Queries for Knn.
        k: Number of neighbours.
        points_row_splits: row_splits for batching points.
        queries_row_splits: row_splits for batching queries.
        return_distances: Whether to return distance with neighbours.

    """
    if points_row_splits.shape[0] != queries_row_splits.shape[0]:
        raise ValueError("KNN(points and queries must have same batch size)")

    points = points.cpu()
    queries = queries.cpu()

    # ml3d knn.
    ans = knn_search(points,
                     queries,
                     k=k,
                     points_row_splits=points_row_splits,
                     queries_row_splits=queries_row_splits,
                     return_distances=True)
    if return_distances:
        return ans.neighbors_index.reshape(
            -1, k).long().cuda(), ans.neighbors_distance.reshape(-1, k).cuda()
    else:
        return ans.neighbors_index.reshape(-1, k).long().cuda()


def interpolation(points,
                  queries,
                  feat,
                  points_row_splits,
                  queries_row_splits,
                  k=3):
    """Interpolation of features with nearest neighbours.

    Args:
        points: Input pointcloud (m, 3).
        queries: Queries for Knn (n, 3).
        feat: features (m, c).
        points_row_splits: row_splits for batching points.
        queries_row_splits: row_splits for batching queries.
        k: Number of neighbours.

    Returns:
        Returns interpolated features (n, c).
    """
    if not (points.is_contiguous and queries.is_contiguous() and
            feat.is_contiguous()):
        raise ValueError("Interpolation (points/queries/feat not contiguous)")
    idx, dist = knn_batch(points,
                          queries,
                          k=k,
                          points_row_splits=points_row_splits,
                          queries_row_splits=queries_row_splits,
                          return_distances=True)  # (n, k), (n, k)

    idx, dist = idx.reshape(-1, k), dist.reshape(-1, k)

    dist_recip = 1.0 / (dist + 1e-8)  # (n, k)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm  # (n, k)

    new_feat = torch.FloatTensor(queries.shape[0],
                                 feat.shape[1]).zero_().to(feat.device)
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat
