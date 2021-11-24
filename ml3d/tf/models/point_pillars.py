import tensorflow as tf
import numpy as np
import pickle
import random

from tqdm import tqdm
import os

from open3d.ml.tf.ops import voxelize

from .base_model_objdet import BaseModel
from ...utils import MODEL

from ..utils.objdet_helper import Anchor3DRangeGenerator, BBoxCoder, multiclass_nms, limit_period, get_paddings_indicator, bbox_overlaps, box3d_to_bev2d
from ..modules.losses.focal_loss import FocalLoss
from ..modules.losses.smooth_L1 import SmoothL1Loss
from ..modules.losses.cross_entropy import CrossEntropyLoss
from ...datasets.utils import ObjdetAugmentation, BEVBox3D
from ...datasets.utils.operations import filter_by_min_points


def unpack(flat_t, counts=None):
    """Converts flat tensor to list of tensors, with length according to
    counts.
    """
    if counts is None:
        return [flat_t]

    data_list = []
    idx0 = 0
    for count in counts:
        idx1 = idx0 + count
        data_list.append(flat_t[idx0:idx1])
        idx0 = idx1
    return data_list


class PointPillars(BaseModel):
    """Object detection model. Based on the PointPillars architecture
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
                 point_cloud_range=[0, -40.0, -3, 70.0, 40.0, 1],
                 classes=['car'],
                 voxelize={},
                 voxel_encoder={},
                 scatter={},
                 backbone={},
                 neck={},
                 head={},
                 loss={},
                 **kwargs):

        super().__init__(name=name,
                         point_cloud_range=point_cloud_range,
                         **kwargs)
        self.point_cloud_range = point_cloud_range
        self.classes = classes
        self.name2lbl = {n: i for i, n in enumerate(classes)}
        self.lbl2name = {i: n for i, n in enumerate(classes)}

        self.voxel_layer = PointPillarsVoxelization(
            point_cloud_range=point_cloud_range, **voxelize)
        self.voxel_encoder = PillarFeatureNet(
            point_cloud_range=point_cloud_range, **voxel_encoder)
        self.middle_encoder = PointPillarsScatter(**scatter)

        self.backbone = SECOND(**backbone)
        self.neck = SECONDFPN(**neck)
        self.bbox_head = Anchor3DHead(num_classes=len(self.classes), **head)

        self.loss_cls = FocalLoss(**loss.get("focal_loss", {}))
        self.loss_bbox = SmoothL1Loss(**loss.get("smooth_l1", {}))
        self.loss_dir = CrossEntropyLoss(**loss.get("cross_entropy", {}))

    def extract_feats(self, points, training=False):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels,
                                            num_points,
                                            coors,
                                            training=training)

        batch_size = int(coors[-1, 0].numpy()) + 1

        x = self.middle_encoder(voxel_features,
                                coors,
                                batch_size,
                                training=training)
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
            coor_pad = tf.pad(coor,
                              paddings,
                              mode='CONSTANT',
                              constant_values=i)
            coors_batch.append(coor_pad)

        coors_batch = tf.concat(coors_batch, axis=0)

        return voxels, num_points, coors_batch

    def call(self, inputs, training=True):
        """Forward pass.

        :param inputs: tuple/list of inputs (points, bboxes, labels, calib)
        :param training: toggle training run
        """
        inputs = unpack(inputs[0], inputs[-2])
        x = self.extract_feats(inputs, training=training)
        outs = self.bbox_head(x, training=training)

        return outs

    def get_optimizer(self, cfg):
        beta1, beta2 = cfg.get('betas', [0.9, 0.99])
        return tf.optimizers.Adam(learning_rate=cfg['lr'],
                                  beta_1=beta1,
                                  beta_2=beta2)

        #used by torch, but doesn't perform well with TF:
        #import tensorflow_addons as tfa
        #beta1, beta2 = cfg.get('betas', [0.9, 0.99])
        #return tfa.optimizers.AdamW(weight_decay=cfg['weight_decay'],
        #                            learning_rate=cfg['lr'],
        #                            beta_1=beta1,
        #                            beta_2=beta2)

    def loss(self, results, inputs, training=True):
        """Computes loss.

        :param results: results of forward pass (scores, bboxes, dirs)
        :param inputs: tuple/list of gt inputs (points, bboxes, labels, calib)
        """
        scores, bboxes, dirs = results

        gt_bboxes, gt_labels = inputs[1:3]
        gt_bboxes = unpack(gt_bboxes, inputs[-1])
        gt_labels = unpack(gt_labels, inputs[-1])

        # generate and filter bboxes
        target_bboxes, target_idx, pos_idx, neg_idx = self.bbox_head.assign_bboxes(
            bboxes, gt_bboxes)

        avg_factor = pos_idx.shape[0]

        # classification loss
        scores = tf.reshape(tf.transpose(scores, (0, 2, 3, 1)),
                            (-1, self.bbox_head.num_classes))
        target_labels = tf.fill((scores.shape[0],),
                                tf.constant(self.bbox_head.num_classes,
                                            dtype=gt_labels[0].dtype))
        gt_label = tf.gather(tf.concat(gt_labels, axis=0), target_idx)
        target_labels = tf.tensor_scatter_nd_update(
            target_labels, tf.expand_dims(pos_idx, axis=-1), gt_label)

        loss_cls = self.loss_cls(
            tf.gather(scores, tf.concat([pos_idx, neg_idx], axis=0)),
            tf.gather(target_labels, tf.concat([pos_idx, neg_idx], axis=0)),
            avg_factor=avg_factor)

        # remove invalid labels
        cond = (gt_label >= 0) & (gt_label < self.bbox_head.num_classes)
        pos_idx = tf.boolean_mask(pos_idx, cond)
        target_idx = tf.boolean_mask(target_idx, cond)
        target_bboxes = tf.boolean_mask(target_bboxes, cond)

        bboxes = tf.reshape(tf.transpose(bboxes, (0, 2, 3, 1)),
                            (-1, self.bbox_head.box_code_size))
        bboxes = tf.gather(bboxes, pos_idx)
        dirs = tf.reshape(tf.transpose(dirs, (0, 2, 3, 1)), (-1, 2))
        dirs = tf.gather(dirs, pos_idx)

        if len(pos_idx) > 0:
            # direction classification loss
            # to discrete bins
            target_dirs = tf.gather(tf.concat(gt_bboxes, axis=0),
                                    target_idx)[:, -1]
            target_dirs = limit_period(target_dirs, 0, 2 * np.pi)
            target_dirs = tf.cast(target_dirs / np.pi, tf.int32) % 2

            loss_dir = self.loss_dir(dirs, target_dirs, avg_factor=avg_factor)

            # bbox loss
            # sinus difference transformation
            r0 = tf.sin(bboxes[:, -1:]) * tf.cos(target_bboxes[:, -1:])
            r1 = tf.cos(bboxes[:, -1:]) * tf.sin(target_bboxes[:, -1:])

            bboxes = tf.concat([bboxes[:, :-1], r0], axis=-1)
            target_bboxes = tf.concat([target_bboxes[:, :-1], r1], axis=-1)

            loss_bbox = self.loss_bbox(bboxes,
                                       target_bboxes,
                                       avg_factor=avg_factor)
        else:
            loss_bbox = tf.reduce_sum(bboxes)
            loss_dir = tf.reduce_sum(dirs)

        return {
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'loss_dir': loss_dir
        }

    def preprocess(self, data, attr):
        points = np.array(data['point'][:, 0:4], dtype=np.float32)

        min_val = np.array(self.point_cloud_range[:3])
        max_val = np.array(self.point_cloud_range[3:])

        points = points[np.where(
            np.all(np.logical_and(points[:, :3] >= min_val,
                                  points[:, :3] < max_val),
                   axis=-1))]

        data['point'] = points

        #Augment data
        if attr['split'] not in ['test', 'testing', 'val', 'validation']:
            data = self.augment_data(data, attr)

        new_data = {'point': data['point'], 'calib': data['calib']}

        if attr['split'] not in ['test', 'testing']:
            new_data['bbox_objs'] = data['bounding_boxes']

        if 'full_point' in data:
            points = np.array(data['full_point'][:, 0:4], dtype=np.float32)

            min_val = np.array(self.point_cloud_range[:3])
            max_val = np.array(self.point_cloud_range[3:])

            points = points[np.where(
                np.all(np.logical_and(points[:, :3] >= min_val,
                                      points[:, :3] < max_val),
                       axis=-1))]

            new_data['full_point'] = points

        return new_data

    def load_gt_database(self, pickle_path, min_points_dict, sample_dict):
        db_boxes = pickle.load(open(pickle_path, 'rb'))

        if min_points_dict is not None:
            db_boxes = filter_by_min_points(db_boxes, min_points_dict)

        db_boxes_dict = {}
        for key in sample_dict.keys():
            db_boxes_dict[key] = []

        for db_box in db_boxes:
            if db_box.label_class in sample_dict.keys():
                db_boxes_dict[db_box.label_class].append(db_box)

        self.db_boxes_dict = db_boxes_dict

    def augment_data(self, data, attr):
        cfg = self.cfg.augment

        if 'ObjectSample' in cfg.keys():
            if not hasattr(self, 'db_boxes_dict'):
                data_path = attr['path']
                # remove tail of path to get root data path
                for _ in range(3):
                    data_path = os.path.split(data_path)[0]
                pickle_path = os.path.join(data_path, 'bboxes.pkl')
                self.load_gt_database(pickle_path, **cfg['ObjectSample'])

            data = ObjdetAugmentation.ObjectSample(
                data,
                db_boxes_dict=self.db_boxes_dict,
                sample_dict=cfg['ObjectSample']['sample_dict'])

        if cfg.get('ObjectRangeFilter', False):
            data = ObjdetAugmentation.ObjectRangeFilter(
                data, self.cfg.point_cloud_range)

        if cfg.get('PointShuffle', False):
            data = ObjdetAugmentation.PointShuffle(data)

        return data

    def transform(self, data, attr):
        points = tf.constant(data['point'], dtype=tf.float32)

        t_data = {'point': points, 'calib': data['calib']}

        if attr['split'] not in ['test', 'testing']:
            t_data['bbox_objs'] = data['bbox_objs']
            t_data['labels'] = tf.constant([
                self.name2lbl.get(bb.label_class, len(self.classes))
                for bb in data['bbox_objs']
            ],
                                           dtype=tf.int32)
            t_data['bboxes'] = tf.constant(
                [bb.to_xyzwhlr() for bb in data['bbox_objs']], dtype=tf.float32)

        return t_data

    def get_batch_gen(self, dataset, steps_per_epoch=None, batch_size=1):

        def batcher():
            count = len(dataset) if steps_per_epoch is None else steps_per_epoch
            for i in np.arange(0, count, batch_size):
                batch = [dataset[i + bi]['data'] for bi in range(batch_size)]
                points = tf.concat([b['point'] for b in batch], axis=0)
                bboxes = tf.concat([
                    b.get('bboxes', tf.zeros((0, 7), dtype=tf.float32))
                    for b in batch
                ],
                                   axis=0)
                labels = tf.concat([
                    b.get('labels', tf.zeros((0,), dtype=tf.int32))
                    for b in batch
                ],
                                   axis=0)

                calib = [
                    tf.constant([
                        b.get('calib', {}).get('world_cam', np.eye(4)),
                        b.get('calib', {}).get('cam_img', np.eye(4))
                    ]) for b in batch
                ]
                count_pts = tf.constant([len(b['point']) for b in batch])
                count_lbs = tf.constant([
                    len(b.get('labels', tf.zeros((0,), dtype=tf.int32)))
                    for b in batch
                ])
                yield (points, bboxes, labels, calib, count_pts, count_lbs)

        gen_func = batcher
        gen_types = (tf.float32, tf.float32, tf.int32, tf.float32, tf.int32,
                     tf.int32)
        gen_shapes = ([None, 4], [None, 7], [None], [batch_size, 2, 4,
                                                     4], [None], [None])

        return gen_func, gen_types, gen_shapes

    def inference_end(self, results, inputs):
        bboxes_b, scores_b, labels_b = self.bbox_head.get_bboxes(*results)

        inference_result = []
        for _calib, _bboxes, _scores, _labels in zip(inputs[3], bboxes_b,
                                                     scores_b, labels_b):
            bboxes = _bboxes.cpu().numpy()
            scores = _scores.cpu().numpy()
            labels = _labels.cpu().numpy()
            inference_result.append([])

            world_cam, cam_img = _calib.numpy()

            for bbox, score, label in zip(bboxes, scores, labels):
                dim = bbox[[3, 5, 4]]
                pos = bbox[:3] + [0, 0, dim[1] / 2]
                yaw = bbox[-1]
                name = self.lbl2name.get(label, "ignore")
                inference_result[-1].append(
                    BEVBox3D(pos, dim, yaw, name, score, world_cam, cam_img))

        return inference_result


MODEL._register_module(PointPillars, 'tf')


class PointPillarsVoxelization(tf.keras.layers.Layer):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points=32,
                 max_voxels=[16000, 40000]):
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
        self.points_range_min = tf.constant(point_cloud_range[:3],
                                            dtype=tf.float32)
        self.points_range_max = tf.constant(point_cloud_range[3:],
                                            dtype=tf.float32)

        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple) or isinstance(max_voxels, list):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = (max_voxels, max_voxels)

    def call(self, points_feats, training=False):
        """Forward function.

        Args:
            points_feats: Tensor with point coordinates and features. The shape
                is [N, 3+C] with N as the number of points and C as the number
                of feature channels.

        Returns:
            (out_voxels, out_coords, out_num_points).
            * out_voxels is a dense list of point coordinates and features for
              each voxel. The shape is [num_voxels, max_num_points, 3+C].
            * out_coords is tensor with the integer voxel coords and shape
              [num_voxels,3]. Note that the order of dims is [z,y,x].
            * out_num_points is a 1D tensor with the number of points for each
              voxel.
        """
        if training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        points = points_feats[:, :3]

        ans = voxelize(
            points,
            tf.convert_to_tensor([0, tf.shape(points)[0]], dtype=tf.int64),
            self.voxel_size, self.points_range_min, self.points_range_max,
            self.max_num_points, max_voxels)

        # prepend row with zeros which maps to index 0 which maps to void points.
        feats = tf.concat([tf.zeros_like(points_feats[0:1, :]), points_feats],
                          axis=0)

        # create raggeed tensor from indices and row splits.
        voxel_point_indices_ragged = tf.RaggedTensor.from_row_splits(
            values=ans.voxel_point_indices,
            row_splits=ans.voxel_point_row_splits)

        # create dense matrix of indices. index 0 maps to the zero vector.
        voxels_point_indices_dense = voxel_point_indices_ragged.to_tensor(
            default_value=-1,
            shape=(voxel_point_indices_ragged.shape[0],
                   self.max_num_points)) + 1

        out_voxels = tf.gather(feats, voxels_point_indices_dense)

        out_coords = tf.concat([
            tf.expand_dims(ans.voxel_coords[:, 2], 1),
            tf.expand_dims(ans.voxel_coords[:, 1], 1),
            tf.expand_dims(ans.voxel_coords[:, 0], 1),
        ],
                               axis=1)

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

        self.norm = tf.keras.layers.BatchNormalization(
            epsilon=1e-3, momentum=0.99, axis=1)  # Pass self.training
        self.linear = tf.keras.layers.Dense(self.units, use_bias=False)

        self.relu = tf.keras.layers.ReLU()

        assert mode in ['max', 'avg']
        self.mode = mode

    #@auto_fp16(apply_to=('inputs'), out_fp32=True)
    def call(self,
             inputs,
             num_voxels=None,
             aligned_distance=None,
             training=False):
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
            x_max = tf.reduce_sum(x, axis=1, keepdims=True) / tf.reshape(
                tf.cast(num_voxels, inputs.dtype), (-1, 1, 1))

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
        points_mean = tf.reduce_sum(
            features[:, :, :3], axis=1, keepdims=True) / tf.reshape(
                tf.cast(num_points, features.dtype), (-1, 1, 1))
        f_cluster = features[:, :, :3] - points_mean
        features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype

        f_center_0 = features[:, :, 0] - (
            tf.expand_dims(tf.cast(coors[:, 3], dtype), 1) * self.vx +
            self.x_offset)
        f_center_1 = features[:, :, 1] - (
            tf.expand_dims(tf.cast(coors[:, 2], dtype), 1) * self.vy +
            self.y_offset)

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
    def call(self, voxel_features, coors, batch_size, training=False):
        """Scatter features of single sample.

        Args:
            voxel_features (tf.Tensor): Voxel features in shape (N, M, C).
            coors (tf.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
            training (bool): Whether we are training or not?
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas_shape = (self.nx * self.ny, self.in_channels)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = tf.boolean_mask(coors, batch_mask)

            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = tf.cast(indices, tf.int64)
            indices = tf.expand_dims(indices, axis=-1)

            voxels = tf.boolean_mask(voxel_features, batch_mask)

            # Now scatter the blob back to the canvas.
            accum = tf.maximum(
                tf.scatter_nd(indices, tf.ones_like(voxels), canvas_shape),
                tf.constant(1.0))
            canvas = tf.scatter_nd(indices, voxels, canvas_shape) / accum
            canvas = tf.transpose(canvas)

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = tf.stack(batch_canvas, axis=0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = tf.reshape(
            batch_canvas, (batch_size, self.in_channels, self.ny, self.nx))

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
            block.add(
                tf.keras.layers.ZeroPadding2D(padding=1,
                                              data_format='channels_first'))
            block.add(tf.keras.layers.Permute((2, 3, 1)))
            block.add(
                tf.keras.layers.Conv2D(filters=out_channels[i],
                                       kernel_size=3,
                                       data_format='channels_last',
                                       use_bias=False,
                                       strides=layer_strides[i]))
            block.add(tf.keras.layers.Permute((3, 1, 2)))
            block.add(
                tf.keras.layers.BatchNormalization(axis=1,
                                                   epsilon=1e-3,
                                                   momentum=0.99))
            block.add(tf.keras.layers.ReLU())

            for j in range(layer_num):
                block.add(
                    tf.keras.layers.ZeroPadding2D(padding=1,
                                                  data_format='channels_first'))
                block.add(tf.keras.layers.Permute((2, 3, 1)))
                block.add(
                    tf.keras.layers.Conv2D(filters=out_channels[i],
                                           kernel_size=3,
                                           data_format='channels_last',
                                           use_bias=False))
                block.add(tf.keras.layers.Permute((3, 1, 2)))
                block.add(
                    tf.keras.layers.BatchNormalization(axis=1,
                                                       epsilon=1e-3,
                                                       momentum=0.99))
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
                upsample_layer = tf.keras.layers.Conv2D(  # TODO : convert to channels last.
                    filters=out_channels[i],
                    kernel_size=stride,
                    data_format='channels_first',
                    use_bias=False,
                    strides=stride,
                    kernel_initializer='he_normal')

            deblock = tf.keras.Sequential()
            deblock.add(upsample_layer)
            deblock.add(
                tf.keras.layers.BatchNormalization(axis=1,
                                                   epsilon=1e-3,
                                                   momentum=0.99))
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

        ups = [
            deblock(x[i], training=training)
            for i, deblock in enumerate(self.deblocks)
        ]

        if len(ups) > 1:
            out = tf.concat(ups, axis=1)
        else:
            out = ups[0]

        return out


class Anchor3DHead(tf.keras.layers.Layer):

    def __init__(self,
                 num_classes=1,
                 in_channels=384,
                 feat_channels=384,
                 nms_pre=100,
                 score_thr=0.1,
                 dir_offset=0,
                 ranges=[[0, -40.0, -3, 70.0, 40.0, 1]],
                 sizes=[[0.6, 1.0, 1.5]],
                 rotations=[0, 1.57],
                 iou_thr=[[0.35, 0.5]]):

        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.nms_pre = nms_pre
        self.score_thr = score_thr
        self.dir_offset = dir_offset
        self.iou_thr = iou_thr

        if len(self.iou_thr) != num_classes:
            assert len(self.iou_thr) == 1
            self.iou_thr = self.iou_thr * num_classes
        assert len(self.iou_thr) == num_classes

        # build anchor generator
        self.anchor_generator = Anchor3DRangeGenerator(ranges=ranges,
                                                       sizes=sizes,
                                                       rotations=rotations)

        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.anchor_generator.num_base_anchors

        # build box coder
        self.bbox_coder = BBoxCoder()
        self.box_code_size = 7

        #Initialize neural network layers of the head.
        self.cls_out_channels = self.num_anchors * self.num_classes

        kernel_init = tf.keras.initializers.RandomNormal(stddev=0.01)
        bias_init = tf.keras.initializers.Constant(
            value=self.bias_init_with_prob(0.01))

        self.conv_cls = tf.keras.layers.Conv2D(self.cls_out_channels,
                                               kernel_size=1,
                                               data_format='channels_last',
                                               kernel_initializer=kernel_init,
                                               bias_initializer=bias_init)

        self.conv_reg = tf.keras.layers.Conv2D(self.num_anchors *
                                               self.box_code_size,
                                               kernel_size=1,
                                               data_format='channels_last',
                                               kernel_initializer=kernel_init)

        self.conv_dir_cls = tf.keras.layers.Conv2D(self.num_anchors * 2,
                                                   kernel_size=1,
                                                   data_format='channels_last')

    @staticmethod
    def bias_init_with_prob(prior_prob):
        """Initialize conv/fc bias value according to giving probablity."""
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))

        return bias_init

    def call(self, x, training=False):
        """Forward function on a feature map.

        Args:
            x (tf.Tensor): Input features.

        Returns:
            tuple[tf.Tensor]: Contain score of each class, bbox \
                regression and direction classification predictions.
        """
        x = tf.transpose(x, perm=[0, 2, 3, 1])

        cls_score = self.conv_cls(x)
        cls_score = tf.transpose(cls_score, perm=[0, 3, 1, 2])

        bbox_pred = self.conv_reg(x)
        bbox_pred = tf.transpose(bbox_pred, perm=[0, 3, 1, 2])

        dir_cls_preds = None
        dir_cls_preds = self.conv_dir_cls(x)
        dir_cls_preds = tf.transpose(dir_cls_preds, perm=[0, 3, 1, 2])

        return cls_score, bbox_pred, dir_cls_preds

    def assign_bboxes(self, pred_bboxes, target_bboxes):
        """Assigns target bboxes to given anchors.

        Args:
            pred_bboxes (tf.Tensor): Bbox predictions (anchors).
            target_bboxes (tf.Tensor): Bbox targets.

        Returns:
            tf.Tensor: Assigned target bboxes for each given anchor.
            tf.Tensor: Flat index of matched targets.
            tf.Tensor: Index of positive matches.
            tf.Tensor: Index of negative matches.
        """
        # compute all anchors
        anchors = [
            self.anchor_generator.grid_anchors(pred_bboxes.shape[-2:])
            for _ in range(len(target_bboxes))
        ]

        # compute size of anchors for each given class
        anchors_count = tf.cast(tf.reduce_prod(anchors[0].shape[:-1]),
                                dtype=tf.int64)
        rot_angles = anchors[0].shape[-2]

        # init the tensors for the final result
        assigned_bboxes, target_idxs, pos_idxs, neg_idxs = [], [], [], []

        def flatten_idx(idx, j):
            """Inject class dimension in the given indices (...

            z * rot_angles + x) --> (.. z * num_classes * rot_angles + j * rot_angles + x)
            """
            z = idx // rot_angles
            x = idx % rot_angles

            return z * self.num_classes * rot_angles + j * rot_angles + x

        idx_off = 0
        for i in range(len(target_bboxes)):
            for j, (neg_th, pos_th) in enumerate(self.iou_thr):
                if target_bboxes[i].shape[0] == 0:
                    continue

                anchors_stride = tf.reshape(anchors[i][..., j, :, :],
                                            (-1, self.box_code_size))

                # compute a fast approximation of IoU
                overlaps = bbox_overlaps(box3d_to_bev2d(target_bboxes[i]),
                                         box3d_to_bev2d(anchors_stride))

                # for each anchor the gt with max IoU
                argmax_overlaps = tf.argmax(overlaps, axis=0)
                max_overlaps = tf.reduce_max(overlaps, axis=0)
                # for each gt the anchor with max IoU
                gt_max_overlaps = tf.reduce_max(overlaps, axis=1)

                pos_idx = max_overlaps >= pos_th
                neg_idx = (max_overlaps >= 0) & (max_overlaps < neg_th)

                # low-quality matching
                for k in range(len(target_bboxes[i])):
                    if gt_max_overlaps[k] >= neg_th:
                        pos_idx = tf.where(overlaps[k, :] == gt_max_overlaps[k],
                                           True, pos_idx)

                pos_idx = tf.where(pos_idx)[:, 0]
                neg_idx = tf.where(neg_idx)[:, 0]
                max_idx = tf.gather(argmax_overlaps, pos_idx)

                # encode bbox for positive matches
                assigned_bboxes.append(
                    self.bbox_coder.encode(tf.gather(anchors_stride, pos_idx),
                                           tf.gather(target_bboxes[i],
                                                     max_idx)))
                target_idxs.append(max_idx + idx_off)

                # store global indices in list
                pos_idx = flatten_idx(pos_idx, j) + i * anchors_count
                neg_idx = flatten_idx(neg_idx, j) + i * anchors_count
                pos_idxs.append(pos_idx)
                neg_idxs.append(neg_idx)

            # compute offset for index computation
            idx_off += len(target_bboxes[i])

        return (tf.concat(assigned_bboxes,
                          axis=0), tf.concat(target_idxs, axis=0),
                tf.concat(pos_idxs, axis=0), tf.concat(neg_idxs, axis=0))

    def get_bboxes(self, cls_scores, bbox_preds, dir_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[tf.Tensor]): Class scores.
            bbox_preds (list[tf.Tensor]): Bbox predictions.
            dir_cls_preds (list[tf.Tensor]): Direction
                class predictions.

        Returns:
            tuple[tf.Tensor]: Prediction results of batches
                (bboxes, scores, labels).
        """
        bboxes, scores, labels = [], [], []
        for cls_score, bbox_pred, dir_pred in zip(cls_scores, bbox_preds,
                                                  dir_preds):

            b, s, l = self.get_bboxes_single(cls_score, bbox_pred, dir_pred)
            bboxes.append(b)
            scores.append(s)
            labels.append(l)
        return bboxes, scores, labels

    def get_bboxes_single(self, cls_scores, bbox_preds, dir_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[tf.Tensor]): Class scores.
            bbox_preds (list[tf.Tensor]): Bbox predictions.
            dir_cls_preds (list[tf.Tensor]): Direction
                class predictions.

        Returns:
            tuple[tf.Tensor]: Prediction results of batches
                (bboxes, scores, labels).
        """
        assert cls_scores.shape[-2:] == bbox_preds.shape[-2:]
        assert cls_scores.shape[-2:] == dir_preds.shape[-2:]

        anchors = self.anchor_generator.grid_anchors(cls_scores.shape[-2:])
        anchors = tf.reshape(anchors, (-1, self.box_code_size))

        dir_preds = tf.reshape(tf.transpose(dir_preds, perm=(1, 2, 0)), (-1, 2))
        dir_scores = tf.math.argmax(dir_preds, axis=-1)

        cls_scores = tf.reshape(tf.transpose(cls_scores, perm=(1, 2, 0)),
                                (-1, self.num_classes))
        scores = tf.sigmoid(cls_scores)

        bbox_preds = tf.reshape(tf.transpose(bbox_preds, perm=(1, 2, 0)),
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
            tf.fill((idxs[i].shape[0],), i) for i in range(self.num_classes)
        ]
        labels = tf.concat(labels, axis=0)

        scores = [
            tf.gather(scores, idxs[i])[:, i] for i in range(self.num_classes)
        ]
        scores = tf.concat(scores, axis=0)

        idxs = tf.concat(idxs, axis=0)
        bboxes = tf.gather(bboxes, idxs)
        dir_scores = tf.gather(dir_scores, idxs)

        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset, 1, np.pi)
            dir_rot = dir_rot + self.dir_offset + np.pi * tf.cast(
                dir_scores, dtype=bboxes.dtype)
            bboxes = tf.concat(
                [bboxes[:, :-1], tf.expand_dims(dir_rot, -1)], axis=-1)

        return bboxes, scores, labels
