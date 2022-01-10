import numpy as np
import tensorflow as tf
import functools
import open3d

from tensorflow.python.framework import ops
from .base_model import BaseModel
from ...utils import MODEL
from ...datasets.augment import SemsegAugmentation

if open3d.core.cuda.device_count() > 0:
    from open3d.ml.tf.ops import trilinear_devoxelize as _trilinear_devoxelize


def trilinear_devoxelize(features, coords, resolution, is_training=True):
    if not open3d.core.cuda.device_count() > 0:
        raise NotImplementedError(
            "Op 'trilinear_devoxelize` not implemented on CPU. Please use a CUDA enabled machine."
        )

    outs, inds, wgts = _trilinear_devoxelize(
        tf.transpose(coords, perm=[0, 2, 1]),
        tf.transpose(features, perm=[0, 4, 1, 2, 3]), resolution, is_training)
    return tf.transpose(outs, perm=[0, 2, 1])


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
        self.point_features = layers

        layers, channels_cloud = create_mlp_components(
            in_channels=channels_point,
            out_channels=[256, 128],
            classifier=False,
            dim=1,
            width_multiplier=width_multiplier)
        self.cloud_features = tf.keras.Sequential(layers=layers)

        layers, _ = create_mlp_components(
            in_channels=(concat_channels_point + channels_cloud),
            out_channels=[512, 0.3, 256, 0.3, num_classes],
            classifier=True,
            dim=2,
            width_multiplier=width_multiplier)
        self.classifier = tf.keras.Sequential(layers=layers)

    def call(self, inputs, training=False):
        """Forward pass for the model.

        Args:
            inputs: A dict object for inputs with following keys
            point (tf.float32): Input pointcloud (B, N,3)
            feat (tf.float32): Input features (B, N, 9)
            training (bool): Whether model is in training phase.

        Returns:
            tf.float32 : probability distribution (B, N, C).

        """
        coords = inputs['point']
        feat = inputs['feat']

        out_features_list = []
        for i in range(len(self.point_features)):
            feat, _ = self.point_features[i]((feat, coords), training=training)
            out_features_list.append(feat)
        # feat: num_batches * num_points * 1024-> num_batches * 1024 -> num_batches * 128

        feat = self.cloud_features(tf.reduce_max(feat, axis=1, keepdims=False),
                                   training=training)

        out_features_list.append(
            tf.transpose(tf.repeat(tf.expand_dims(feat, -1),
                                   coords.shape[1],
                                   axis=-1),
                         perm=[0, 2, 1]))

        out = self.classifier(tf.concat(out_features_list, axis=-1),
                              training=training)

        return out

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

        choices = np.random.choice(
            points.shape[0],
            self.cfg.num_points,
            replace=(points.shape[0] < self.cfg.num_points))
        points = points[choices]
        feat = feat[choices]
        labels = labels[choices]

        data = {}
        data['point'] = points
        data['feat'] = feat
        data['label'] = labels

        return data

    def transform(self, points, feat, labels):
        """Transform function for the point cloud and features.

        This function is called after preprocess method by dataset generator.
        It consists of mapping data to dict.

        Args:
            points: Input pointcloud.
            feat: Input features.
            labels: Input labels.

        Returns:
            Returns dictionary data with keys
            (point, feat, label).

        """
        return {'point': points, 'feat': feat, 'label': labels}

    def get_batch_gen(self, dataset, steps_per_epoch=None, batch_size=1):

        def gen():
            iters = dataset.num_pc

            for idx in range(iters):
                data, attr = dataset.read_data(idx)
                yield data['point'], data['feat'], data['label']

        gen_func = gen
        gen_types = (tf.float32, tf.float32, tf.int32)
        gen_shapes = ([None, 3], [None, 9], [None])

        return gen_func, gen_types, gen_shapes

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

    def get_loss(self, sem_seg_loss, results, inputs):
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
        labels = tf.reshape(inputs['label'], -1)
        results = tf.reshape(results, (-1, results.shape[-1]))

        scores, labels = sem_seg_loss.filter_valid_label(results, labels)
        loss = sem_seg_loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg_pipeline['learning_rate'])

        return optimizer


MODEL._register_module(PVCNN, 'tf')


class SE3d(tf.keras.layers.Layer):
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
        fc = tf.keras.Sequential()
        fc.add(
            tf.keras.layers.Dense(channel // reduction,
                                  use_bias=False,
                                  input_shape=(channel,)))
        fc.add(tf.keras.layers.ReLU())
        fc.add(tf.keras.layers.Dense(channel, use_bias=False))
        fc.add(tf.keras.layers.Sigmoid())

        self.fc = fc

    def call(self, inputs, training):
        """Forward call for SE3d

        Args:
            inputs: Input features.
            training (bool): Whether model is in training phase.

        Returns:
            Transformed features.

        """
        return inputs * self.fc(inputs.mean(-1).mean(-1).mean(-1)).view(
            inputs.shape[0], inputs.shape[1], 1, 1, 1)


def _linear_bn_relu(in_channels, out_channels):
    """Layer combining Linear, BatchNorm and ReLU Block."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(out_channels, input_shape=(in_channels,)))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(tf.keras.layers.ReLU())

    return model


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
        return tf.keras.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(tf.keras.layers.Dropout(rate=oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(tf.keras.layers.Dense(out_channels[-1]))
        else:
            layers.append(
                _linear_bn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(
                tf.keras.layers.Conv1D(filters=out_channels[-1], kernel_size=1))
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


class SharedMLP(tf.keras.layers.Layer):
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
            conv = tf.keras.layers.Conv1D
            bn = tf.keras.layers.BatchNormalization
        elif dim == 2:
            conv = tf.keras.layers.Conv2D
            bn = tf.keras.layers.BatchNormalization
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        layers = []
        for oc in out_channels:
            layers.extend([
                conv(filters=oc, kernel_size=1),
                bn(momentum=0.9, epsilon=1e-5),
                tf.keras.layers.ReLU(),
            ])
            in_channels = oc
        self.layers = tf.keras.Sequential(layers=layers)

    def call(self, inputs, training):
        """Forward pass for SharedMLP

        Args:
            inputs: features or a list of features.
            training (bool): Whether model is in training phase.

        Returns:
            Transforms first features in a list.

        """
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0], training=training), *inputs[1:])
        else:
            return self.layers(inputs, training=training)


class PVConv(tf.keras.layers.Layer):
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
            tf.keras.layers.Conv3D(filters=out_channels,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   padding="same"),  # kernel_size // 2
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-4),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv3D(filters=out_channels,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   padding="same"),
            tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-4),
            tf.keras.layers.LeakyReLU(0.1),
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))

        self.voxel_layers = tf.keras.Sequential(layers=voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)

    def call(self, inputs, training):
        """Forward pass for PVConv.

        Args:
            inputs: tuple of features and coordinates.
            training (bool): Whether model is in training phase.

        Returns:
            Fused features consists of point features and
            voxel_features.

        """
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features,
                                                         coords,
                                                         training=training)
        voxel_features = self.voxel_layers(voxel_features, training=training)
        voxel_features = trilinear_devoxelize(voxel_features, voxel_coords,
                                              self.resolution, training)

        point_features = self.point_features(features, training=training)
        fused_features = voxel_features + point_features

        return fused_features, coords


def avg_voxelize(feat, coords, r):
    """Voxelize points and returns a voxel_grid with
    mean of features lying in same voxel.

    Args:
        feat: Input features (B, N, 3).
        coords: Input coordinates (B, N, C).
        r (int): Resolution of voxel grid.

    Returns:
        voxel grid (B, r, r, r, C)

    """
    shape = tf.constant([feat.shape[0], r, r, r, feat.shape[2]])
    batch_id = tf.repeat(tf.range(0, feat.shape[0]), feat.shape[1])
    batch_id = tf.reshape(batch_id, (-1, 1))
    coords = tf.reshape(coords, (-1, 3))
    coords = tf.concat([batch_id, coords], 1)
    feat = tf.reshape(feat, (-1, shape[-1]))

    grid = tf.scatter_nd(coords, feat, shape)

    count = tf.scatter_nd(coords, tf.ones(feat.shape), shape)
    count = tf.where(count == 0, 1, count)

    return grid / count


class Voxelization(tf.keras.layers.Layer):
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

    def call(self, features, coords, training):
        """Forward pass for Voxelization.

        Args:
            features: Input features.
            coords: Input coordinates.
            training (bool): Whether model is in training phase.

        Returns:
            Voxel grid of features (B, C, r, r, r)

        """
        coords = tf.stop_gradient(coords)
        norm_coords = coords - tf.reduce_mean(coords, axis=1, keepdims=True)

        if self.normalize:
            norm_coords = norm_coords / (
                tf.reduce_max(tf.norm(norm_coords, axis=2, keepdims=True),
                              axis=1,
                              keepdims=True) * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = tf.clip_by_value(norm_coords * self.r, 0, self.r - 1)
        vox_coords = tf.cast(tf.round(norm_coords), tf.int32)

        return avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        """Extra representation of module."""
        return 'resolution={}{}'.format(
            self.r,
            ', normalized eps = {}'.format(self.eps) if self.normalize else '')
