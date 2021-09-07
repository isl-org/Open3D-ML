import numpy as np
import torch
import torch.nn as nn
import functools

from .base_model import BaseModel
from ...utils import MODEL
from .utils import trilinear_devoxelize
from ..modules.losses import filter_valid_label
from ...datasets.augment import SemsegAugmentation


class PVCNN(BaseModel):
    blocks = ((64, 1, 32), (64, 2, 16), (128, 1, 16), (1024, 1, None))

    def __init__(self,
                 name='PVCNN',
                 device="cuda",
                 num_classes=13,
                 num_points=40960,
                 extra_feature_channels=6,
                 width_multiplier=1,
                 voxel_resolution_multiplier=1,
                 batcher='DefaultBatcher',
                 augment=None,
                 **kwargs):
        super(PVCNN, self).__init__(
            name=name,
            device=device,
            num_classes=num_classes,
            num_points=num_points,
            extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
            batcher=batcher,
            augment=augment,
            **kwargs)
        cfg = self.cfg
        self.device = device
        self.augmenter = SemsegAugmentation(cfg.augment)
        self.in_channels = extra_feature_channels + 3

        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks,
            in_channels=self.in_channels,
            with_se=False,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.point_features = nn.ModuleList(layers)

        layers, channels_cloud = create_mlp_components(
            in_channels=channels_point,
            out_channels=[256, 128],
            classifier=False,
            dim=1,
            width_multiplier=width_multiplier)
        self.cloud_features = nn.Sequential(*layers)

        layers, _ = create_mlp_components(
            in_channels=(concat_channels_point + channels_cloud),
            out_channels=[512, 0.3, 256, 0.3, num_classes],
            classifier=True,
            dim=2,
            width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

    def forward(self, inputs):
        coords = inputs['point'].to(self.device)
        feat = inputs['feat'].to(self.device)

        # coords = inputs[:, :3, :]
        out_features_list = []
        for i in range(len(self.point_features)):
            feat, _ = self.point_features[i]((feat, coords))
            out_features_list.append(feat)
        # feat: num_batches * 1024 * num_points -> num_batches * 1024 -> num_batches * 128
        feat = self.cloud_features(feat.max(dim=-1, keepdim=False).values)
        out_features_list.append(
            feat.unsqueeze(-1).repeat([1, 1, coords.size(-1)]))
        out = self.classifier(torch.cat(out_features_list, dim=1))
        return out.transpose(1, 2)

    def preprocess(self, data, attr):
        points = np.array(data['point'], dtype=np.float32)

        if 'label' not in data or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data or data['feat'] is None:
            feat = points.copy()
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        if attr['split'] in ['training', 'train']:
            points, feat, labels = self.augmenter.augment(
                points, feat, labels, self.cfg.get('augment', None))

        points -= np.min(points, 0)

        feat = feat / 255.0

        max_points_x = np.max(points[:, 0])
        max_points_y = np.max(points[:, 1])
        max_points_z = np.max(points[:, 2])

        x, y, z = np.split(points, (1, 2), axis=-1)
        norm_x = x / max_points_x
        norm_y = y / max_points_y
        norm_z = z / max_points_z

        feat = np.concatenate([x, y, z, feat, norm_x, norm_y, norm_z], axis=-1)

        choices = np.random.choice(
            points.shape[0],
            self.cfg.num_points,
            replace=(points.shape[0] < self.cfg.num_points))
        points = points[choices].transpose()
        feat = feat[choices].transpose()
        labels = labels[choices]

        data = {}
        data['point'] = points
        data['feat'] = feat
        data['label'] = labels

        return data

    def transform(self, data, attr):
        data['point'] = torch.from_numpy(data['point'])
        data['feat'] = torch.from_numpy(data['feat'])
        data['label'] = torch.from_numpy(data['label'])

        return data

    def update_probs(self, inputs, results, test_probs, test_labels):
        result = results.reshape(-1, self.cfg.num_classes)
        probs = torch.nn.functional.softmax(result, dim=-1).cpu().data.numpy()
        labels = np.argmax(probs, 1)

        self.trans_point_sampler(patchwise=False)

        return probs, labels

    def inference_begin(self, data):
        data = self.preprocess(data, {'split': 'test'})
        data['batch_lengths'] = [data['point'].shape[0]]
        data = self.transform(data, {})

        self.inference_input = data

    def inference_preprocess(self):
        return self.inference_input

    def inference_end(self, inputs, results):
        results = torch.reshape(results, (-1, self.cfg.num_classes))

        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()

        probs = np.reshape(results, [-1, self.cfg.num_classes])

        pred_l = np.argmax(probs, 1)

        return {'predict_labels': pred_l, 'predict_scores': probs}

    def get_loss(self, Loss, results, inputs, device):
        """Calculate the loss on output of the model.

        Attributes:
            Loss: Object of type `SemSegLoss`.
            results: Output of the model.
            inputs: Input of the model.
            device: device(cpu or cuda).
        
        Returns:
            Returns loss, labels and scores.
        """
        cfg = self.cfg
        labels = inputs['data']['label'].reshape(-1,)
        results = results.reshape(-1, results.shape[-1])

        scores, labels = filter_valid_label(results, labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, cfg_pipeline.scheduler_gamma)

        return optimizer, scheduler


MODEL._register_module(PVCNN, 'torch')


class SE3d(nn.Module):

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, inputs):
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(
            inputs.shape[0], inputs.shape[1], 1, 1, 1)


def _linear_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels),
                         nn.BatchNorm1d(out_channels), nn.ReLU(True))


def create_mlp_components(in_channels,
                          out_channels,
                          classifier=False,
                          dim=2,
                          width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_bn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and
                                  out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(
                _linear_bn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(blocks,
                               in_channels,
                               with_se=False,
                               normalize=True,
                               eps=0,
                               width_multiplier=1,
                               voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    layers, concat_channels = [], 0
    for out_channels, num_blocks, voxel_resolution in blocks:
        out_channels = int(r * out_channels)
        if voxel_resolution is None:
            block = SharedMLP
        else:
            block = functools.partial(PVConv,
                                      kernel_size=3,
                                      resolution=int(vr * voxel_resolution),
                                      with_se=with_se,
                                      normalize=normalize,
                                      eps=eps)
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
            concat_channels += out_channels
    return layers, in_channels, concat_channels


class SharedMLP(nn.Module):

    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.extend([
                conv(in_channels, oc, 1),
                bn(oc),
                nn.ReLU(True),
            ])
            in_channels = oc
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)


class PVConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 resolution,
                 with_se=False,
                 normalize=True,
                 eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution,
                                         normalize=normalize,
                                         eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels,
                      out_channels,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = trilinear_devoxelize(voxel_features, voxel_coords,
                                              self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords


def avg_voxelize(feat, coords, r):
    """
    coords (B, 3, N)
    feat (B, C, N)
    return (B, C, r, r, r)
    """
    coords = coords.to(torch.int64)
    batch_size = feat.shape[0]
    dim = feat.shape[1]
    grid = torch.zeros((batch_size, dim, r, r, r)).to(feat.device)

    batch_id = torch.from_numpy(np.arange(batch_size).reshape(-1, 1)).to(
        feat.device)
    hash = batch_id * r * r * r + coords[:,
                                         0, :] * r * r + coords[:,
                                                                1, :] * r + coords[:,
                                                                                   2, :]
    hash = hash.reshape(-1,).to(feat.device)

    for i in range(0, dim):
        grid_ = torch.zeros(batch_size * r * r * r,
                            device=feat.device).scatter_add_(
                                0, hash, feat[:, i, :].reshape(-1,)).reshape(
                                    batch_size, r, r, r)
        grid[:, i] = grid_
    count = torch.zeros(batch_size * r * r * r,
                        device=feat.device).scatter_add_(
                            0, hash, torch.ones_like(feat[:, 0, :].reshape(
                                -1,))).reshape(batch_size, 1, r, r,
                                               r).clamp(min=1)
    count[count == 0] = 1
    grid = grid / count

    return grid


class Voxelization(nn.Module):

    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(
                dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 +
                                         self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)

        return avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(
            self.r,
            ', normalized eps = {}'.format(self.eps) if self.normalize else '')


if __name__ == '__main__':
    # vox = Voxelization(10)
    # feat = torch.rand(1, 10, 3)
    # coords = torch.rand(1, 10, 3) * 100

    # res = vox(feat, coords)
    # print(vox)
    feat = torch.Tensor([[1., 2, 3], [3, 4., 5], [4, 6, 7]])
    coords = torch.IntTensor([[0, 1, 1], [0, 1, 1], [2, 1, 0]])
    res = avg_voxelize(feat, coords, 3)
    print(res)
    res.backward()
