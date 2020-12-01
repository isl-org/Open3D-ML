import tensorflow as tf
import numpy as np
import random
import open3d.ml.tf as ml3d

from tqdm import tqdm

from ...vis.boundingbox import BoundingBox3D

from .base_model import BaseModel
from ...utils import MODEL

from ..utils.objdet_helper import Anchor3DRangeGenerator, BBoxCoder, multiclass_nms, limit_period, get_paddings_indicator

class PointPillars(BaseModel):
    """Object detection model. 
    Based on the PointPillars architecture 
    https://github.com/nutonomy/second.pytorch.

    Args:
        name (string): Name of model.
            Default to "PointPillars".
        voxel_size: voxel edge lengths with format [x, y, z].
        point_cloud_range: The valid range of point coordinates as
            [x_min, y_min, z_min, x_max, y_max, z_max].
        voxelize: Config of PointPillarsVoxelization module.
        voxelize_encoder: Config of PillarFeatureNet module.
        scatter: Config of PointPillarsScatter module.
        backbone: Config of backbone module (SECOND).
        neck: Config of neck module (SECONDFPN).
        head: Config of anchor head module.
    """

    def __init__(self,
                 name="PointPillars",
                 voxel_size=[0.16, 0.16, 4],
                 point_cloud_range=[0, -40.0, -3, 70.0, 40.0, 1],
                 voxelize={},
                 voxel_encoder={},
                 scatter={},
                 backbone={},
                 neck={},
                 head={},
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self.point_cloud_range = point_cloud_range

        self.voxel_layer = PointPillarsVoxelization(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            **voxelize)
        self.voxel_encoder = PillarFeatureNet(
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            **voxel_encoder)
        self.middle_encoder = PointPillarsScatter(**scatter)

        self.backbone = SECOND(**backbone)
        self.neck = SECONDFPN(**neck)
        self.bbox_head = Anchor3DHead(**head)

    def extract_feats(self, points, training=False):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors, training=training)

        batch_size = int(coors[-1, 0].numpy()) + 1

        x = self.middle_encoder(voxel_features, coors, batch_size, training=training)
        x = self.backbone(x, training=training)
        x = self.neck(x, training=training)

        return x

    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)

        voxels = tf.concat(voxels, axis=0)
        num_points = tf.concat(num_points, axis=0)

        coors_batch = []
        for i, coor in enumerate(coors):
            paddings = [[0, 0] for i in range(len(coor.shape))]
            paddings[-1] = [1, 0]
            coor_pad = tf.pad(coor, paddings, mode='CONSTANT', constant_values=i)
            coors_batch.append(coor_pad)

        coors_batch = tf.concat(coors_batch, axis=0)

        return voxels, num_points, coors_batch


    def call(self, inputs, training=True):
        x = self.extract_feats(inputs, training=training)
        outs = self.bbox_head(x, training=training)

        return outs


    def get_optimizer(self, cfg_pipeline):
        raise NotImplementedError

    def get_loss(self, Loss, results, inputs):
        raise NotImplementedError

    def preprocess(self, data, attr):
        return data

    def transform(self, data, attr):
        points = np.array(data['point'][:, 0:4], dtype=np.float32)

        min_val = np.array(self.point_cloud_range[:3])
        max_val = np.array(self.point_cloud_range[3:])

        points = points[np.where(
            np.all(np.logical_and(points[:, :3] >= min_val,
                                  points[:, :3] < max_val),
                   axis=-1))]

        if 'bounding_boxes' not in data.keys(
        ) or data['bounding_boxes'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = data['bounding_boxes']

        if 'feat' not in data.keys() or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        calib = data['calib']

        data = dict()
        data['point'] = points
        data['feat'] = feat
        data['calib'] = calib
        data['bounding_boxes'] = labels

        return data

    def inference_begin(self, data):
        self.inference_data = data

    def inference_preprocess(self):
        data = tf.convert_to_tensor([self.inference_data["point"]], dtype=np.float32)

        return {"data": data}

    def inference_end(self, inputs, results):
        bboxes_b, scores_b, labels_b = self.bbox_head.get_bboxes(*results)

        self.inference_result = []

        for _bboxes, _scores, _labels in zip(bboxes_b, scores_b, labels_b):
            bboxes = _bboxes.numpy()
            scores = _scores.numpy()
            labels = _labels.numpy()
            self.inference_result.append([])

            for bbox, score, label in zip(bboxes, scores, labels):
                yaw = bbox[-1]
                cos = np.cos(yaw)
                sin = np.sin(yaw)

                front = np.array((sin, cos, 0))
                left = np.array((-cos, sin, 0))
                up = np.array((0, 0, 1))

                dim = bbox[[3, 5, 4]]
                pos = bbox[:3] + [0, 0, dim[1] / 2]

                self.inference_result[-1].append(
                    BoundingBox3D(pos, front, up, left, dim, label, score))

        return True


MODEL._register_module(PointPillars, 'tf')


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

        # create raggeed tensor from indices and row splits.
        voxel_point_indices_ragged = tf.RaggedTensor.from_row_splits(
            values=ans.voxel_point_indices,
            row_splits=ans.voxel_point_row_splits
        )

        # create dense matrix of indices. index 0 maps to the zero vector.
        voxels_point_indices_dense = voxel_point_indices_ragged.to_tensor(
            default_value=-1,
            shape=(voxel_point_indices_ragged.shape[0], self.max_num_points)
            ) + 1

        out_voxels = tf.gather(feats, voxels_point_indices_dense)

        out_coords = tf.concat([
            tf.expand_dims(ans.voxel_coords[:, 2], 1),
            tf.expand_dims(ans.voxel_coords[:, 1], 1),
            tf.expand_dims(ans.voxel_coords[:, 0], 1),
            ], axis=1)

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
        self._name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.99)  # Pass self.training
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
            x_max = tf.reduce_max(x, axis=1, keepdims=True)
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
        points_mean = tf.reduce_sum(features[:, :, :3], axis=1, keepdims=True) / tf.reshape(tf.cast(num_points, features.dtype), (-1, 1, 1))
        f_cluster = features[:, :, :3] - points_mean
        features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype

        f_center_0 = features[:, :, 0] - (tf.expand_dims(tf.cast(coors[:, 3], dtype), 1) * self.vx + self.x_offset)
        f_center_1 = features[:, :, 1] - (tf.expand_dims(tf.cast(coors[:, 2], dtype), 1) * self.vy + self.y_offset)

        f_center = tf.stack((f_center_0, f_center_1), axis=2)

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
            features = pfn(features, num_points, training=training)

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
        self.out_shape = output_shape
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
            this_coors = tf.boolean_mask(coors, batch_mask)

            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = tf.cast(indices, tf.int64)

            voxels = tf.boolean_mask(voxel_features, batch_mask)
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


class SECOND(tf.keras.layers.Layer):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
    """

    def __init__(self,
                 in_channels=64,
                 out_channels=[64, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2]):
        super(SECOND, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = tf.keras.Sequential()
            block.add(tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_first'))
            block.add(tf.keras.layers.Conv2D(filters=out_channels[i], kernel_size=3, data_format='channels_first', use_bias=False, strides=layer_strides[i]))
            block.add(tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-3, momentum=0.99))
            block.add(tf.keras.layers.ReLU())

            for j in range(layer_num):
                block.add(tf.keras.layers.ZeroPadding2D(padding=1, data_format='channels_first'))
                block.add(tf.keras.layers.Conv2D(filters=out_channels[i], kernel_size=3, data_format='channels_first', use_bias=False))
                block.add(tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-3, momentum=0.99))
                block.add(tf.keras.layers.ReLU())

            blocks.append(block)

        self.blocks = blocks

    def call(self, x, training=False):
        """Forward function.

        Args:
            x (tf.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[tf.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, training=training)
            outs.append(x)
        return tuple(outs)


class SECONDFPN(tf.keras.layers.Layer):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[64, 128, 256],
                 out_channels=[128, 128, 128],
                 upsample_strides=[1, 2, 4],
                 use_conv_for_no_stride=False):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, epsilon=1e-3, affine=True)
        super(SECONDFPN, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = tf.keras.layers.Conv2DTranspose(
                    filters=out_channel,
                    kernel_size=upsample_strides[i],
                    strides=upsample_strides[i],
                    use_bias=False,
                    data_format='channels_first',
                )
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = tf.keras.layers.Conv2D(
                    filters=out_channels[i],
                    kernel_size=stride,
                    data_format='channels_first',
                    use_bias=False,
                    strides=stride
                    )

            deblock = tf.keras.Sequential()
            deblock.add(upsample_layer)
            deblock.add(tf.keras.layers.BatchNormalization(axis=1, epsilon=1e-3, momentum=0.99))
            deblock.add(tf.keras.layers.ReLU())

            deblocks.append(deblock)

        self.deblocks = deblocks

    #@auto_fp16()
    def call(self, x, training=False):
        """Forward function.

        Args:
            x (tf.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            tf.Tensor: Feature maps.
        """
        assert len(x) == len(self.in_channels)

        ups = [deblock(x[i], training=training) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = tf.concat(ups, axis=1)
        else:
            out = ups[0]

        return out


class Anchor3DHead(tf.keras.layers.Layer):
    def __init__(self, 
                 num_classes=3, 
                 in_channels=384, 
                 feat_channels=384,
                 nms_pre=100,
                 score_thr=0.1,
                 ranges=[
                    [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                    [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                    [0, -39.68, -1.78, 70.4, 39.68, -1.78],
                 ],
                 sizes=[[0.6, 0.8, 1.73],
                        [0.6, 1.76, 1.73],
                        [1.6, 3.9, 1.56]],
                 rotations=[0, 1.57]):

        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.nms_pre = nms_pre
        self.score_thr = score_thr

        # build anchor generator
        self.anchor_generator = Anchor3DRangeGenerator(
            ranges=ranges,
            sizes=sizes,
            rotations=rotations)

        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.anchor_generator.num_base_anchors

        # build box coder
        self.bbox_coder = BBoxCoder()
        self.box_code_size = 7

        self.fp16_enabled = False

        #Initialize neural network layers of the head.
        self.cls_out_channels = self.num_anchors * self.num_classes

        self.conv_cls = tf.keras.layers.Conv2D(self.cls_out_channels, kernel_size=1, data_format='channels_first')
        self.conv_reg = tf.keras.layers.Conv2D(self.num_anchors * self.box_code_size, kernel_size=1, data_format='channels_first')

        self.conv_dir_cls = tf.keras.layers.Conv2D(self.num_anchors * 2, kernel_size=1, data_format='channels_first')


    def call(self, x, training=False):
        """Forward function on a feature map.

        Args:
            x (tf.Tensor): Input features.

        Returns:
            tuple[tf.Tensor]: Contain score of each class, bbox \
                regression and direction classification predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        dir_cls_preds = self.conv_dir_cls(x)

        return cls_score, bbox_pred, dir_cls_preds

    def get_bboxes(self, cls_scores, bbox_preds, dir_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Class scores.
            bbox_preds (list[torch.Tensor]): Bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Direction
                class predictions.

        Returns:
            tuple[torch.Tensor]: Prediction results of batches 
                (bboxes, scores, labels).
        """
        bboxes, scores, labels = [], [], []
        for cls_score, bbox_pred, dir_pred in zip(cls_scores, bbox_preds, dir_preds):

            b, s, l = self.get_bboxes_single(cls_score, bbox_pred, dir_pred)
            bboxes.append(b)
            scores.append(s)
            labels.append(l)
        return bboxes, scores, labels

    def get_bboxes_single(self, cls_scores, bbox_preds, dir_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Class scores.
            bbox_preds (list[torch.Tensor]): Bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Direction
                class predictions.

        Returns:
            tuple[torch.Tensor]: Prediction results of batches 
                (bboxes, scores, labels).
        """
        assert cls_scores.shape[-2:] == bbox_preds.shape[-2:]
        assert cls_scores.shape[-2:] == dir_preds.shape[-2:]

        anchors = self.anchor_generator.grid_anchors(cls_scores.shape[-2:])
        anchors = tf.reshape(anchors, (-1, self.box_code_size))

        dir_preds = tf.reshape(
            tf.transpose(dir_preds, perm=(1, 2, 0)),
            (-1, 2))
        dir_scores = tf.math.argmax(dir_preds, axis=-1)

        cls_scores = tf.reshape(
            tf.transpose(cls_scores, perm=(1, 2, 0)),
            (-1, self.num_classes))
        scores = tf.sigmoid(cls_scores)

        bbox_preds = tf.reshape(
            tf.transpose(bbox_preds, perm=(1, 2, 0)),
            (-1, self.box_code_size))

        if scores.shape[0] > self.nms_pre:
            max_scores = tf.reduce_max(scores, axis=1)
            _, topk_inds = tf.math.top_k(max_scores, self.nms_pre)
            anchors = tf.gather(anchors, topk_inds)
            bbox_preds = tf.gather(bbox_preds, topk_inds)
            scores = tf.gather(scores, topk_inds)
            dir_scores = tf.gather(dir_scores, topk_inds)

        bboxes = self.bbox_coder.decode(anchors, bbox_preds)

        idxs = multiclass_nms(bboxes, scores, self.score_thr)

        labels = [
            tf.fill((idxs[i].shape[0],), i)
            for i in range(self.num_classes)
        ]
        labels = tf.concat(labels, axis=0)

        scores = [tf.gather(scores, idxs[i])[:, i] for i in range(self.num_classes)]
        scores = tf.concat(scores, axis=0)

        idxs = tf.concat(idxs, axis=0)
        bboxes = tf.gather(bboxes, idxs)
        dir_scores = tf.gather(dir_scores, idxs)

        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6], 1, np.pi)
            dir_rot = dir_rot + np.pi * tf.cast(dir_scores, dtype=bboxes.dtype)
            bboxes = tf.concat([
                bboxes[:,:-1], 
                tf.expand_dims(dir_rot, -1)
            ], axis=-1)

        return bboxes, scores, labels
