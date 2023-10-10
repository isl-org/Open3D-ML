import numpy as np
import torch
import torch.nn as nn
import functools
import open3d
from torch.autograd import Function

from .base_model import BaseModel
from ...utils import MODEL
from ..modules.losses import filter_valid_label
from ...datasets.augment import SemsegAugmentation

if open3d.core.cuda.device_count() > 0:
    from open3d.ml.torch.ops import trilinear_devoxelize_forward, trilinear_devoxelize_backward


class TrilinearDevoxelization(Function):

    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """Forward pass for the Op.

        Args:
            ctx: torch Autograd context.
            coords: the coordinates of points, FloatTensor[B, 3, N]
            features: FloatTensor[B, C, R, R, R]
            resolution: int, the voxel resolution.
            is_training: bool, training mode.
        
        Returns:
            torch.FloatTensor: devoxelized features (B, C, N)

        """
        B, C = features.shape[:2]
        features = features.contiguous()
        coords = coords.contiguous()
        outs, inds, wgts = trilinear_devoxelize_forward(resolution, is_training,
                                                        coords, features)
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for the Op.

        Args:
            ctx: torch Autograd context
            grad_output: gradient of outputs, FloatTensor[B, C, N]

        Returns:
            torch.FloatTensor: gradient of inputs (B, C, R, R, R)

        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = trilinear_devoxelize_backward(grad_output.contiguous(),
                                                    inds, wgts, ctx.r)
        return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.r,
                                ctx.r, ctx.r), None, None, None


trilinear_devoxelize = TrilinearDevoxelization.apply


class PVCNN(BaseModel):
    """Semantic Segmentation model. Based on Point Voxel Convolutions.
    https://arxiv.org/abs/1907.03739

    Uses PointNet architecture with separate Point and Voxel processing.

    Attributes:
        name: Name of model.
          Default to "PVCNN".
        num_classes: Number of classes.
        num_points: Number of points to sample per pointcloud.
        extra_feature_channels: Number of extra features.
          Default to 6 (RGB + Coordinate norms).
        batcher: Batching method for dataloader.
        augment: dictionary for augmentation.
    """

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
        """Forward pass for the model.

        Args:
            inputs: A dict object for inputs with following keys
                point (torch.float32): Input pointcloud (B, 3, N)
                feat (torch.float32): Input features (B, 9, N)

        Returns:
            torch.float32 : probability distribution (B, N, C).

        """
        coords = inputs['point'].to(self.device)
        feat = inputs['feat'].to(self.device)

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
        """Data preprocessing function.

        This function is called before training to preprocess the data from a
        dataset. It consists of subsampling and normalizing the pointcloud and
        creating new features.

        Args:
            data: A sample from the dataset.
            attr: The corresponding attributes.

        Returns:
            Returns the preprocessed data

        """
        # If num_workers > 0, use new RNG with unique seed for each thread.
        # Else, use default RNG.
        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(
                torch.utils.data.get_worker_info().seed +
                torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng

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

        feat = feat / 255.0  # Normalize to [0, 1]

        max_points_x = np.max(points[:, 0])
        max_points_y = np.max(points[:, 1])
        max_points_z = np.max(points[:, 2])

        x, y, z = np.split(points, (1, 2), axis=-1)
        norm_x = x / max_points_x
        norm_y = y / max_points_y
        norm_z = z / max_points_z

        feat = np.concatenate([x, y, z, feat, norm_x, norm_y, norm_z], axis=-1)

        choices = rng.choice(points.shape[0],
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
        """Transform function for the point cloud and features.

        This function is called after preprocess method. It consists
        of converting numpy arrays to torch Tensors.

        Args:
            data: A sample from the dataset.
            attr: The corresponding attributes.

        Returns:
            Returns dictionary data with keys
            (point, feat, label).

        """
        data['point'] = torch.from_numpy(data['point'])
        data['feat'] = torch.from_numpy(data['feat'])
        data['label'] = torch.from_numpy(data['label'])

        return data

    def update_probs(self, inputs, results, test_probs):
        result = results.reshape(-1, self.cfg.num_classes)
        probs = torch.nn.functional.softmax(result, dim=-1).cpu().data.numpy()

        self.trans_point_sampler(patchwise=False)

        return probs

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

    def get_loss(self, sem_seg_loss, results, inputs, device):
        """Calculate the loss on output of the model.

        Attributes:
            sem_seg_loss: Object of type `SemSegLoss`.
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

        loss = sem_seg_loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **cfg_pipeline.optimizer)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, cfg_pipeline.scheduler_gamma)

        return optimizer, scheduler


MODEL._register_module(PVCNN, 'torch')


class SE3d(nn.Module):
    """Extra Sequential Dense layers to be used to increase
    model complexity.
    """

    def __init__(self, channel, reduction=8):
        """Constructor for SE3d module.

        Args:
            channel: Number of channels in the input layer.
            reduction: Factor of channels in second layer.

        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, inputs):
        """Forward call for SE3d

        Args:
            inputs: Input features.

        Returns:
            Transformed features.

        """
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(
            inputs.shape[0], inputs.shape[1], 1, 1, 1)


def _linear_bn_relu(in_channels, out_channels):
    """Layer combining Linear, BatchNorm and ReLU Block."""
    return nn.Sequential(nn.Linear(in_channels, out_channels),
                         nn.BatchNorm1d(out_channels), nn.ReLU(True))


def create_mlp_components(in_channels,
                          out_channels,
                          classifier=False,
                          dim=2,
                          width_multiplier=1):
    """Creates multiple layered components. For each output channel,
    it creates Dense layers with Dropout.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        classifier: Whether the layer is classifier(appears at the end).
        dim: Dimension
        width_multiplier: factor by which neurons expands in intermediate layers.

    Returns:
        A List of layers.

    """
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
                               eps=1e-6,
                               width_multiplier=1,
                               voxel_resolution_multiplier=1):
    """Creates pointnet components. For each output channel,
    it comprises of PVConv or SharedMLP layers.

    Args:
        blocks: list of (out_channels, num_blocks, voxel_resolution).
        in_channels: Number of input channels.
        with_se: Whether to use extra dense layers in each block.
        normalize: Whether to normalize pointcloud before voxelization.
        eps: Epsilon for voxelization.
        width_multiplier: factor by which neurons expands in intermediate layers.
        voxel_resolution_multiplier: Factor by which voxel resolution expands.

    Returns:
        A List of layers, input_channels, and concat_channels

    """
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
    """SharedMLP Module, comprising Conv2d, BatchNorm and ReLU blocks."""

    def __init__(self, in_channels, out_channels, dim=1):
        """Constructor for SharedMLP Block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            dim: Input dimension

        """
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
        """Forward pass for SharedMLP

        Args:
            inputs: features or a list of features.

        Returns:
            Transforms first features in a list.

        """
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)


class PVConv(nn.Module):
    """Point Voxel Convolution module. Consisting of 3D Convolutions
    for voxelized pointcloud, and SharedMLP blocks for point features.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 resolution,
                 with_se=False,
                 normalize=True,
                 eps=1e-6):
        """Constructor for PVConv module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: kernel size for Conv3D.
            resolution: Resolution of the voxel grid.
            with_se: Whether to use extra dense layers in each block.
            normalize: Whether to normalize pointcloud before voxelization.
            eps: Epsilon for voxelization.

        """
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
        """Forward pass for PVConv.

        Args:
            inputs: tuple of features and coordinates.

        Returns:
            Fused features consists of point features and
            voxel_features.

        """
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = trilinear_devoxelize(voxel_features, voxel_coords,
                                              self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords


def avg_voxelize(feat, coords, r):
    """Voxelize points and returns a voxel_grid with
    mean of features lying in same voxel.

    Args:
        feat: Input features (B, 3, N).
        coords: Input coordinates (B, C, N).
        r (int): Resolution of voxel grid.

    Returns:
        voxel grid (B, C, r, r, r)

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
    """Voxelization module. Normalize the coordinates and
    returns voxel_grid with mean of features lying in same
    voxel.
    """

    def __init__(self, resolution, normalize=True, eps=1e-6):
        """Constructor of Voxelization module.

        Args:
            resolution (int): Resolution of the voxel grid.
            normalize (bool): Whether to normalize coordinates.
            eps (float): Small epsilon to avoid nan.

        """
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        """Forward pass for Voxelization.

        Args:
            features: Input features.
            coords: Input coordinates.

        Returns:
            Voxel grid of features (B, C, r, r, r)

        """
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
        """Extra representation of module."""
        return 'resolution={}{}'.format(
            self.r,
            ', normalized eps = {}'.format(self.eps) if self.normalize else '')
