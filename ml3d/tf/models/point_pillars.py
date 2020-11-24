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
    def forward(self, inputs, num_voxels=None, aligned_distance=None, training=False):
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
