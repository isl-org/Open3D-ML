import tensorflow as tf
import numpy as np
import random
import open3d.ml.tf as ml3d

from tqdm import tqdm

from .base_model import BaseModel
from ...utils import MODEL


class PointPillarsVoxelization(tf.keras.layers.Layer):
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points=32,
                 max_voxels=(16000, 40000)):
        """Voxelization layer for the PointPillars model.

        Args:
            voxel_size: voxel edge lengths with format [x, y, z].
            point_cloud_range: The valid range of point coordinates as
                [x_min, y_min, z_min, x_max, y_max, z_max].
            max_num_points: The maximum number of points per voxel.
            max_voxels: The maximum number of voxels. May be a tuple with
                values for training and testing.
        """
        super().__init__()
        self.voxel_size = tf.constant(voxel_size, dtype=tf.float32)
        self.point_cloud_range = point_cloud_range
        self.points_range_min = tf.constant(point_cloud_range[:3], dtype=tf.float32)
        self.points_range_max = tf.constant(point_cloud_range[3:], dtype=tf.float32)

        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = (max_voxels, max_voxels)

    def call(self, points_feats, training=False):
        """Forward function

        Args:
            points_feats: Tensor with point coordinates and features. The shape
                is [N, 3+C] with N as the number of points and C as the number 
                of feature channels.
        Returns:
            (out_voxels, out_coords, out_num_points).
            - out_voxels is a dense list of point coordinates and features for 
              each voxel. The shape is [num_voxels, max_num_points, 3+C].
            - out_coords is tensor with the integer voxel coords and shape
              [num_voxels,3]. Note that the order of dims is [z,y,x].
            - out_num_points is a 1D tensor with the number of points for each
              voxel.
        """
        if training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        points = points_feats[:, :3]

        ans = ml3d.ops.voxelize(points, self.voxel_size, self.points_range_min,
                                self.points_range_max, self.max_num_points,
                                max_voxels)

        # prepend row with zeros which maps to index 0 which maps to void points.
        feats = tf.concat([tf.zeros_like(points_feats[0:1, :]), points_feats], axis=0)

        # create dense matrix of indices. index 0 maps to the zero vector.
        voxels_point_indices_dense = ml3d.ops.ragged_to_dense(
            ans.voxel_point_indices, ans.voxel_point_row_splits,
            self.max_num_points, tf.constant(-1)) + 1

        out_voxels = feats[voxels_point_indices_dense]
        out_coords = ans.voxel_coords[:, [2, 1, 0]].contiguous()
        out_num_points = ans.voxel_point_row_splits[
            1:] - ans.voxel_point_row_splits[:-1]

        return out_voxels, out_coords, out_num_points


class PFNLayer(tf.keras.layers.Layer):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self, in_channels, out_channels, last_layer=False, mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = tf.keras.layers.BatchNormalization(eps=1e-3, momentum=0.99)  # Pass self.training
        self.linear = tf.keras.layers.Dense(self.units, use_bias=False)

        self.relu = tf.keras.layers.ReLU()

        assert mode in ['max', 'avg']
        self.mode = mode

    #@auto_fp16(apply_to=('inputs'), out_fp32=True)
    def call(self, inputs, num_voxels=None, aligned_distance=None, training=False):
        """Forward function.

        Args:
            inputs (tf.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (tf.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (tf.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            tf.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)
        x = self.norm(tf.transpose(x, perm=[0, 2, 1]), training=training)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = tf.matmul(x, tf.expand_dims(aligned_distance, -1))
            x_max = tf.reduce_max(x, axis=1, keepdims=True)[0]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = tf.matmul(x, tf.expand_dims(aligned_distance, -1))
            x_max = tf.reduce_sum(x, axis=1, keepdims=True) / tf.reshape(tf.cast(num_voxels, inputs.dtype), (-1, 1, 1))

        if self.last_vfe:
            return x_max
        else:
            x_repeat = tf.repeat(x_max, inputs.shape[1], axis=1)
            x_concatenated = tf.concat([x, x_repeat], axis=2)
            return x_concatenated


class PillarFeatureNet(tf.keras.layers.Layer):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64,),
                 voxel_size=(0.16, 0.16, 4),
                 point_cloud_range=(0, -39.68, -3, 69.12, 39.68, 1)):

        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0

        # with cluster center (+3) + with voxel center (+2)
        in_channels += 5

        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(in_filters,
                         out_filters,
                         last_layer=last_layer,
                         mode='max'))
        self.pfn_layers = pfn_layers

        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    #@force_fp32(out_fp16=True)
    def call(self, features, num_points, coors, training=False):
        """Forward function.

        Args:
            features (tf.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (tf.Tensor): Number of points in each pillar.
            coors (tf.Tensor): Coordinates of each voxel.

        Returns:
            tf.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        points_mean = tf.reduce_sum(features[:, :, :3], axis=1, keepdims=True) / tf.reshape(tf.cast(num_points, features.dtype), -1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype

        f_center = features[:, :, :2]
        f_center[:, :, 0] = f_center[:, :, 0] - (tf.expand_dims(tf.cast(coors[:, 3], dtype), 1) * self.vx + self.x_offset)
        f_center[:, :, 1] = f_center[:, :, 1] - (tf.expand_dims(tf.cast(coors[:, 2], dtype), 1) * self.vy + self.y_offset)

        features_ls.append(f_center)

        # Combine together feature decorations
        features = tf.concat(features_ls, axis=-1)

        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = tf.cast(tf.expand_dims(mask, -1), dtype)

        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points, training)

        return tf.squeeze(features)


class PointPillarsScatter(tf.keras.layers.Layer):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels=64, output_shape=[496, 432]):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

    #@auto_fp16(apply_to=('voxel_features', ))
    def call(self, voxel_features, coors, batch_size=None, training=False):
        """Forward function to scatter features."""
        if batch_size is not None:
            return self.forward_batch(voxel_features, coors, batch_size, training)
        else:
            return self.forward_single(voxel_features, coors, training)

    def forward_single(self, voxel_features, coors, training=False):
        """Scatter features of single sample.

        Args:
            voxel_features (tf.Tensor): Voxel features in shape (N, M, C).
            coors (tf.Tensor): Coordinates of each voxel.
                The first column indicates the sample ID.
        """
        # Create the canvas for this sample
        canvas = tf.zeros((self.in_channels, self.nx * self.ny), dtype=voxel_features.dtype)

        indices = coors[:, 1] * self.nx + coors[:, 2]
        indices = tf.cast(indices, tf.int64)
        voxels = tf.transpose(voxel_features)

        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels

        # Undo the column stacking to final 4-dim tensor
        canvas = tf.reshape(canvas, (1, self.in_channels, self.ny, self.nx))

        return [canvas]

    def forward_batch(self, voxel_features, coors, batch_size, training=False):
        """Scatter features of single sample.

        Args:
            voxel_features (tf.Tensor): Voxel features in shape (N, M, C).
            coors (tf.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = tf.zeros((self.in_channels, self.nx * self.ny), dtype=voxel_features.dtype)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = tf.cast(indices, tf.int64)
            voxels = voxel_features[batch_mask, :]
            voxels = tf.transpose(voxels)

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = tf.stack(batch_canvas, axis=0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = tf.reshape(batch_canvas, (batch_size, self.in_channels, self.ny, self.nx))

        return batch_canvas