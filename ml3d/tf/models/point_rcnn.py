import tensorflow as tf
import numpy as np

from .base_model_objdet import BaseModel
from ..modules.losses.smooth_L1 import SmoothL1Loss
from ..modules.losses.focal_loss import FocalLoss
from ..modules.losses.cross_entropy import CrossEntropyLoss
from ..modules.pointnet import Pointnet2MSG, PointnetSAModule
from ..utils.objdet_helper import xywhr_to_xyxyr
from open3d.ml.tf.ops import nms
from ..utils.tf_utils import gen_CNN
from ...datasets.utils import BEVBox3D, DataProcessing
from ...datasets.utils.operations import points_in_box
from ...datasets.augment import ObjdetAugmentation

from ...utils import MODEL
from ..modules.schedulers import OneCycleScheduler

from ..utils.roipool3d import roipool3d_utils
from ...metrics import iou_3d


class PointRCNN(BaseModel):
    """Object detection model. Based on the PoinRCNN architecture
    https://github.com/sshaoshuai/PointRCNN.

    The network is not trainable end-to-end, it requires pre-training of the RPN
    module, followed by training of the RCNN module.  For this the mode must be
    set to 'RPN', with this, the network only outputs intermediate results.  If
    the RPN module is trained, the mode can be set to 'RCNN' (default), with
    this, the second module can be trained and the output are the final
    predictions.

    For inference use the 'RCNN' mode.

    Args:
        name (string): Name of model.
            Default to "PointRCNN".
        device (string): 'cuda' or 'cpu'.
            Default to 'cuda'.
        classes (string[]): List of classes used for object detection:
            Default to ['Car'].
        score_thres (float): Min confindence score for prediction.
            Default to 0.3.
        npoints (int): Number of processed input points.
            Default to 16384.
        rpn (dict): Config of RPN module.
            Default to {}.
        rcnn (dict): Config of RCNN module.
            Default to {}.
        mode (string): Execution mode, 'RPN' or 'RCNN'.
            Default to 'RCNN'.
    """

    def __init__(self,
                 name="PointRCNN",
                 classes=['Car'],
                 score_thres=0.3,
                 npoints=16384,
                 rpn={},
                 rcnn={},
                 mode="RCNN",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        assert mode == "RPN" or mode == "RCNN"
        self.mode = mode

        self.augmenter = ObjdetAugmentation(self.cfg.augment)
        self.npoints = npoints
        self.classes = classes
        self.name2lbl = {n: i for i, n in enumerate(classes)}
        self.lbl2name = {i: n for i, n in enumerate(classes)}
        self.score_thres = score_thres

        self.rpn = RPN(**rpn)
        self.rcnn = RCNN(num_classes=len(self.classes), **rcnn)

        if self.mode == "RCNN":
            self.rpn.trainable = False
        else:
            self.rcnn.trainable = False

    def call(self, inputs, training=True):
        cls_score, reg_score, backbone_xyz, backbone_features = self.rpn(
            inputs[0], training=(self.mode == "RPN" and training))

        if self.mode != "RPN":
            cls_score = tf.stop_gradient(cls_score)
            reg_score = tf.stop_gradient(reg_score)
            backbone_xyz = tf.stop_gradient(backbone_xyz)
            backbone_features = tf.stop_gradient(backbone_features)

        rpn_scores_raw = tf.stop_gradient(cls_score[:, :, 0])
        rois, _ = self.rpn.proposal_layer(rpn_scores_raw,
                                          reg_score,
                                          backbone_xyz,
                                          training=self.mode == "RPN" and
                                          training)  # (B, M, 7)
        rois = tf.stop_gradient(rois)

        output = {"rois": rois, "cls": cls_score, "reg": reg_score}

        if self.mode == "RCNN":
            rpn_scores_norm = tf.sigmoid(rpn_scores_raw)

            seg_mask = tf.cast((rpn_scores_norm > self.score_thres), tf.float32)
            pts_depth = tf.norm(backbone_xyz, ord=2, axis=2)

            seg_mask = tf.stop_gradient(seg_mask)
            pts_depth = tf.stop_gradient(pts_depth)

            gt_boxes = None
            if training or self.mode == "RPN":
                gt_boxes = inputs[1]

            output = self.rcnn(rois,
                               gt_boxes,
                               backbone_xyz,
                               tf.transpose(backbone_features, (0, 2, 1)),
                               seg_mask,
                               pts_depth,
                               training=training)

        return output

    def get_optimizer(self, cfg):

        beta1, beta2 = cfg.get('betas', [0.9, 0.99])
        lr_scheduler = OneCycleScheduler(40800, cfg.lr, cfg.div_factor)

        optimizer = tf.optimizers.Adam(learning_rate=lr_scheduler,
                                       beta_1=beta1,
                                       beta_2=beta2)

        return optimizer

    def loss(self, results, inputs, training=True):
        if self.mode == "RPN":
            return self.rpn.loss(results, inputs)
        else:
            if not training:
                return {"loss": tf.constant(0.0)}
            return self.rcnn.loss(results, inputs)

    def filter_objects(self, bbox_objs):
        """Filter objects based on classes to train.

        Args:
            bbox_objs: Bounding box objects from dataset class.

        Returns:
            Filtered bounding box objects.

        """
        filtered = []
        for bb in bbox_objs:
            if bb.label_class in self.classes:
                filtered.append(bb)
        return filtered

    def preprocess(self, data, attr):
        if attr['split'] in ['train', 'training']:
            data = self.augmenter.augment(data, attr)

        data['bounding_boxes'] = self.filter_objects(data['bounding_boxes'])

        # remove intensity
        points = np.array(data['point'][..., :3], dtype=np.float32)
        calib = data['calib']

        # transform in cam space
        points = DataProcessing.world2cam(points, calib['world_cam'])

        new_data = {'point': points, 'calib': calib}

        # bounding_boxes are objects of type BEVBox3D. It is renamed to
        # bbox_objs to clarify them as objects and not matrix of type [N, 7].
        if attr['split'] not in ['test', 'testing']:
            new_data['bbox_objs'] = data['bounding_boxes']

        return new_data

    @staticmethod
    def generate_rpn_training_labels(points, bboxes, bboxes_world, calib=None):
        """Generates labels for RPN network.

        Classifies each point as foreground/background based on points inside bbox.
        We don't train on ambiguous points which are just outside bounding boxes(calculated
        by `extended_boxes`).
        Also computes regression labels for bounding box proposals(in bounding box frame).

        Args:
            points: Input pointcloud.
            bboxes: bounding boxes in camera frame.
            bboxes_world: bounding boxes in world frame.
            calib: Calibration file for cam_to_world matrix.

        Returns:
            Classification and Regression labels.

        """
        cls_label = np.zeros((points.shape[0]), dtype=np.int32)
        reg_label = np.zeros((points.shape[0], 7),
                             dtype=np.float32)  # dx, dy, dz, ry, h, w, l

        if len(bboxes) == 0:
            return cls_label, reg_label

        pts_idx = points_in_box(points.copy(),
                                bboxes_world,
                                camera_frame=True,
                                cam_world=DataProcessing.invT(
                                    calib['world_cam']))

        # enlarge the bbox3d, ignore nearby points
        extended_boxes = bboxes_world.copy()
        # Enlarge box by 0.4m (from PointRCNN paper).
        extended_boxes[3:6] += 0.4
        # Decrease z coordinate, as z_center is at bottom face of box.
        extended_boxes[:, 2] -= 0.2

        pts_idx_ext = points_in_box(points.copy(),
                                    extended_boxes,
                                    camera_frame=True,
                                    cam_world=DataProcessing.invT(
                                        calib['world_cam']))

        for k in range(bboxes.shape[0]):
            fg_pt_flag = pts_idx[:, k]
            fg_pts_rect = points[fg_pt_flag]
            cls_label[fg_pt_flag] = 1

            fg_enlarge_flag = pts_idx_ext[:, k]
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            center3d = bboxes[k][0:3].copy()  # (x, y, z)
            center3d[1] -= bboxes[k][3] / 2
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect

            # size and angle encoding
            reg_label[fg_pt_flag, 3] = bboxes[k][3]  # h
            reg_label[fg_pt_flag, 4] = bboxes[k][4]  # w
            reg_label[fg_pt_flag, 5] = bboxes[k][5]  # l
            reg_label[fg_pt_flag, 6] = bboxes[k][6]  # ry

        return cls_label, reg_label

    def transform(self, data, attr):
        points = data['point']

        if attr['split'] not in ['test', 'testing']:  #, 'val', 'validation']:
            if self.npoints < len(points):
                pts_depth = points[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs,
                                                    self.npoints -
                                                    len(far_idxs_choice),
                                                    replace=False)

                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                if self.npoints > len(points):
                    extra_choice = np.random.choice(choice,
                                                    self.npoints - len(points),
                                                    replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            points = points[choice, :]

        t_data = {'point': points, 'calib': data['calib']}

        if attr['split'] not in ['test', 'testing']:
            labels = []
            bboxes = []
            bboxes_world = []
            if len(data['bbox_objs']) != 0:
                labels = np.stack([
                    self.name2lbl.get(bb.label_class, len(self.classes))
                    for bb in data['bbox_objs']
                ])

                bboxes = np.stack([bb.to_camera() for bb in data['bbox_objs']
                                  ])  # Camera frame.
                bboxes_world = np.stack(
                    [bb.to_xyzwhlr() for bb in data['bbox_objs']])

            if self.mode == "RPN":
                labels, bboxes = PointRCNN.generate_rpn_training_labels(
                    points, bboxes, bboxes_world, data['calib'])
            t_data['labels'] = np.array(labels)
            t_data['bbox_objs'] = data['bbox_objs']  # Objects of type BEVBox3D.
            if attr['split'] in ['train', 'training'] or self.mode == "RPN":
                t_data['bboxes'] = bboxes

        return t_data

    def inference_end(self, results, inputs):
        if self.mode == 'RPN':
            return [[]]

        roi_boxes3d = results['rois']  # (B, M, 7)
        batch_size = roi_boxes3d.shape[0]

        rcnn_cls = tf.reshape(results['cls'],
                              (batch_size, -1, results['cls'].shape[1]))
        rcnn_reg = tf.reshape(results['reg'],
                              (batch_size, -1, results['reg'].shape[1]))

        pred_boxes3d, rcnn_cls = self.rcnn.proposal_layer(rcnn_cls,
                                                          rcnn_reg,
                                                          roi_boxes3d,
                                                          training=False)

        inference_result = []
        for calib, bboxes, scores in zip(inputs[3], pred_boxes3d, rcnn_cls):
            # scoring
            if scores.shape[-1] == 1:
                scores = tf.sigmoid(scores)
                labels = tf.cast(scores < self.score_thres, tf.int64)
            else:
                labels = tf.argmax(scores)
                scores = tf.nn.softmax(scores, axis=0)
                scores = scores[labels]

            fltr = tf.reshape(scores > self.score_thres, (-1))
            bboxes = bboxes[fltr]
            labels = labels[fltr]
            scores = scores[fltr]

            bboxes = bboxes.numpy()
            scores = scores.numpy()
            labels = labels.numpy()
            inference_result.append([])

            world_cam, cam_img = calib.numpy()

            for bbox, score, label in zip(bboxes, scores, labels):
                pos = bbox[:3]
                dim = bbox[[4, 3, 5]]
                # transform into world space
                pos = DataProcessing.cam2world(pos.reshape((1, -1)),
                                               world_cam).flatten()
                pos = pos + [0, 0, dim[1] / 2]
                yaw = bbox[-1]

                inference_result[-1].append(
                    BEVBox3D(pos, dim, yaw, label[0], score, world_cam,
                             cam_img))

        return inference_result

    def get_batch_gen(self, dataset, steps_per_epoch=None, batch_size=1):

        def batcher():
            count = len(dataset) if steps_per_epoch is None else steps_per_epoch
            for i in np.arange(0, count, batch_size):
                batch = [dataset[i + bi]['data'] for bi in range(batch_size)]
                points = tf.stack([b['point'] for b in batch], axis=0)

                bboxes = []
                for b in batch:
                    if ('bboxes' not in b) or len(b['bboxes']) == 0:
                        bboxes.append(tf.zeros((0, 7), dtype=tf.float32))
                    else:
                        bboxes.append(b['bboxes'])
                max_gt = 0
                for bbox in bboxes:
                    max_gt = max(max_gt, bbox.shape[0])
                pad_bboxes = np.zeros((len(bboxes), max_gt, 7),
                                      dtype=np.float32)
                for j in range(len(bboxes)):
                    pad_bboxes[j, :bboxes[j].shape[0], :] = bboxes[j]
                bboxes = tf.constant(pad_bboxes)

                labels = [
                    b.get('labels', tf.zeros((0,), dtype=tf.int32))
                    for b in batch
                ]
                max_lab = 0
                for lab in labels:
                    max_lab = max(max_lab, lab.shape[0])

                if 'labels' in batch[
                        0] and labels[0].shape[0] != points.shape[1]:
                    pad_labels = np.ones(
                        (len(labels), max_lab), dtype=np.int32) * (-1)
                    for j in range(len(labels)):
                        pad_labels[j, :labels[j].shape[0]] = labels[j]
                    labels = tf.constant(pad_labels)

                else:
                    labels = tf.stack(labels, axis=0)

                calib = [
                    tf.constant([
                        b.get('calib', {}).get('world_cam', np.eye(4)),
                        b.get('calib', {}).get('cam_img', np.eye(4))
                    ]) for b in batch
                ]
                yield (points, bboxes, labels, calib)

        gen_func = batcher
        gen_types = (tf.float32, tf.float32, tf.int32, tf.float32)
        gen_shapes = ([batch_size, None, 3], [batch_size, None,
                                              7], [batch_size,
                                                   None], [batch_size, 2, 4, 4])

        return gen_func, gen_types, gen_shapes


MODEL._register_module(PointRCNN, 'tf')


def get_reg_loss(pred_reg,
                 reg_label,
                 loc_scope,
                 loc_bin_size,
                 num_head_bin,
                 anchor_size,
                 get_xz_fine=True,
                 get_y_by_bin=False,
                 loc_y_scope=0.5,
                 loc_y_bin_size=0.25,
                 get_ry_fine=False):
    """Bin-based 3D bounding boxes regression loss. See
    https://arxiv.org/abs/1812.04244 for more details.

    Args:
        pred_reg: (N, C)
        reg_label: (N, 7) [dx, dy, dz, h, w, l, ry]
        loc_scope: Constant
        loc_bin_size: Constant
        num_head_bin: Constant
        anchor_size: (N, 3) or (3)
        get_xz_fine: Whether to get fine xz loss.
        get_y_by_bin: Whether to divide y coordinate into bin.
        loc_y_scope: Scope length for y coordinate.
        loc_y_bin_size: Bin size for classifying y coordinate.
        get_ry_fine: Whether to use fine yaw loss.
    """
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    reg_loss_dict = {}
    loc_loss = 0

    # xz localization loss
    x_offset_label, y_offset_label, z_offset_label = reg_label[:,
                                                               0], reg_label[:,
                                                                             1], reg_label[:,
                                                                                           2]
    x_shift = tf.clip_by_value(x_offset_label + loc_scope, 0,
                               loc_scope * 2 - 1e-3)
    z_shift = tf.clip_by_value(z_offset_label + loc_scope, 0,
                               loc_scope * 2 - 1e-3)
    x_bin_label = tf.cast(tf.floor(x_shift / loc_bin_size), tf.int64)
    z_bin_label = tf.cast(tf.floor(z_shift / loc_bin_size), tf.int64)

    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r

    loss_x_bin = CrossEntropyLoss()(pred_reg[:, x_bin_l:x_bin_r], x_bin_label)
    loss_z_bin = CrossEntropyLoss()(pred_reg[:, z_bin_l:z_bin_r], z_bin_label)
    reg_loss_dict['loss_x_bin'] = loss_x_bin.numpy()
    reg_loss_dict['loss_z_bin'] = loss_z_bin.numpy()
    loc_loss += loss_x_bin + loss_z_bin

    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_label = x_shift - (
            tf.cast(x_bin_label, tf.float32) * loc_bin_size + loc_bin_size / 2)
        z_res_label = z_shift - (
            tf.cast(z_bin_label, tf.float32) * loc_bin_size + loc_bin_size / 2)
        x_res_norm_label = x_res_label / loc_bin_size
        z_res_norm_label = z_res_label / loc_bin_size

        x_bin_onehot = tf.one_hot(x_bin_label, per_loc_bin_num)
        z_bin_onehot = tf.one_hot(z_bin_label, per_loc_bin_num)

        loss_x_res = SmoothL1Loss()(tf.reduce_sum(pred_reg[:, x_res_l:x_res_r] *
                                                  x_bin_onehot,
                                                  axis=1), x_res_norm_label)
        loss_z_res = SmoothL1Loss()(tf.reduce_sum(pred_reg[:, z_res_l:z_res_r] *
                                                  z_bin_onehot,
                                                  axis=1), z_res_norm_label)
        reg_loss_dict['loss_x_res'] = loss_x_res.numpy()
        reg_loss_dict['loss_z_res'] = loss_z_res.numpy()
        loc_loss += loss_x_res + loss_z_res

    # y localization loss
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_shift = tf.clip_by_value(y_offset_label + loc_y_scope, 0,
                                   loc_y_scope * 2 - 1e-3)
        y_bin_label = tf.cast(tf.floor(y_shift / loc_y_bin_size), tf.int64)
        y_res_label = y_shift - (tf.cast(y_bin_label, tf.float32) *
                                 loc_y_bin_size + loc_y_bin_size / 2)
        y_res_norm_label = y_res_label / loc_y_bin_size

        y_bin_onehot = tf.one_hot(y_bin_label, loc_y_bin_num)

        loss_y_bin = CrossEntropyLoss()(pred_reg[:, y_bin_l:y_bin_r],
                                        y_bin_label)
        loss_y_res = SmoothL1Loss()(tf.reduce_sum(pred_reg[:, y_res_l:y_res_r] *
                                                  y_bin_onehot,
                                                  axis=1), y_res_norm_label)

        reg_loss_dict['loss_y_bin'] = loss_y_bin.numpy()
        reg_loss_dict['loss_y_res'] = loss_y_res.numpy()

        loc_loss += loss_y_bin + loss_y_res
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        loss_y_offset = SmoothL1Loss()(tf.reduce_sum(
            pred_reg[:, y_offset_l:y_offset_r], axis=1), y_offset_label)
        reg_loss_dict['loss_y_offset'] = loss_y_offset.numpy()
        loc_loss += loss_y_offset

    # angle loss
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_label = reg_label[:, 6]

    if get_ry_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin

        ry_label = ry_label % (2 * np.pi)  # 0 ~ 2pi
        ry_label = tf.where((ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5),
                            (ry_label + np.pi) % (2 * np.pi),
                            ry_label)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        shift_angle = (ry_label + np.pi * 0.5) % (2 * np.pi)  # (0 ~ pi)

        shift_angle = tf.clip_by_value(shift_angle - np.pi * 0.25, 1e-3,
                                       np.pi * 0.5 - 1e-3)  # (0, pi/2)

        # bin center is (5, 10, 15, ..., 85)
        ry_bin_label = tf.cast(tf.floor(shift_angle / angle_per_class),
                               tf.int64)
        ry_res_label = shift_angle - (tf.cast(ry_bin_label, tf.float32) *
                                      angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    else:
        # divide 2pi into several bins
        angle_per_class = (2 * np.pi) / num_head_bin
        heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi

        shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
        ry_bin_label = tf.cast(tf.floor(shift_angle / angle_per_class),
                               tf.int64)
        ry_res_label = shift_angle - (tf.cast(ry_bin_label, tf.float32) *
                                      angle_per_class + angle_per_class / 2)
        ry_res_norm_label = ry_res_label / (angle_per_class / 2)

    ry_bin_onehot = tf.one_hot(ry_bin_label, num_head_bin)
    loss_ry_bin = CrossEntropyLoss()(pred_reg[:, ry_bin_l:ry_bin_r],
                                     ry_bin_label)
    loss_ry_res = SmoothL1Loss()(tf.reduce_sum(pred_reg[:, ry_res_l:ry_res_r] *
                                               ry_bin_onehot,
                                               axis=1), ry_res_norm_label)

    reg_loss_dict['loss_ry_bin'] = loss_ry_bin.numpy()
    reg_loss_dict['loss_ry_res'] = loss_ry_res.numpy()
    angle_loss = loss_ry_bin + loss_ry_res

    # size loss
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert pred_reg.shape[1] == size_res_r, '%d vs %d' % (pred_reg.shape[1],
                                                          size_res_r)

    size_res_norm_label = (reg_label[:, 3:6] - anchor_size) / anchor_size
    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    size_loss = SmoothL1Loss()(size_res_norm, size_res_norm_label)

    # Total regression loss
    reg_loss_dict['loss_loc'] = loc_loss
    reg_loss_dict['loss_angle'] = angle_loss
    reg_loss_dict['loss_size'] = size_loss

    return loc_loss, angle_loss, size_loss, reg_loss_dict


class RPN(tf.keras.layers.Layer):

    def __init__(self,
                 backbone={},
                 cls_in_ch=128,
                 cls_out_ch=[128],
                 reg_in_ch=128,
                 reg_out_ch=[128],
                 db_ratio=0.5,
                 head={},
                 focal_loss={},
                 loss_weight=[1.0, 1.0],
                 **kwargs):

        super().__init__()

        # backbone
        self.backbone = Pointnet2MSG(**backbone)
        self.proposal_layer = ProposalLayer(**head)

        # classification branch
        layers = []
        for i in range(len(cls_out_ch)):
            layers.extend([
                tf.keras.layers.Conv1D(cls_out_ch[i],
                                       1,
                                       use_bias=False,
                                       data_format="channels_first"),
                tf.keras.layers.BatchNormalization(axis=1,
                                                   momentum=0.9,
                                                   epsilon=1e-05),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(db_ratio)
            ])
        layers.append(
            tf.keras.layers.Conv1D(
                1,
                1,
                use_bias=True,
                bias_initializer=tf.keras.initializers.Constant(-np.log(
                    (1 - 0.01) / 0.01)),
                data_format="channels_first"))

        self.cls_blocks = tf.keras.Sequential(layers)

        # regression branch
        per_loc_bin_num = int(self.proposal_layer.loc_scope /
                              self.proposal_layer.loc_bin_size) * 2
        if self.proposal_layer.loc_xz_fine:
            reg_channel = per_loc_bin_num * 4 + self.proposal_layer.num_head_bin * 2 + 3
        else:
            reg_channel = per_loc_bin_num * 2 + self.proposal_layer.num_head_bin * 2 + 3
        reg_channel = reg_channel + 1  # reg y

        layers = []
        for i in range(len(reg_out_ch)):
            layers.extend([
                tf.keras.layers.Conv1D(reg_out_ch[i],
                                       1,
                                       use_bias=False,
                                       data_format="channels_first"),
                tf.keras.layers.BatchNormalization(axis=1,
                                                   momentum=0.9,
                                                   epsilon=1e-05),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(db_ratio)
            ])
        layers.append(
            tf.keras.layers.Conv1D(
                reg_channel,
                1,
                use_bias=True,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.001),
                data_format="channels_first"))

        self.reg_blocks = tf.keras.Sequential(layers)

        self.loss_cls = FocalLoss(**focal_loss)
        self.loss_weight = loss_weight

    def call(self, x, training=True):
        backbone_xyz, backbone_features = self.backbone(
            x, training=training)  # (B, N, 3), (B, C, N)

        rpn_cls = tf.transpose(
            self.cls_blocks(backbone_features, training=training),
            (0, 2, 1))  # (B, N, 1)
        rpn_reg = tf.transpose(
            self.reg_blocks(backbone_features, training=training),
            (0, 2, 1))  # (B, N, C)

        return rpn_cls, rpn_reg, backbone_xyz, backbone_features

    def loss(self, results, inputs):
        rpn_cls = results['cls']
        rpn_reg = results['reg']

        rpn_reg_label = inputs[1]
        rpn_cls_label = inputs[2]

        rpn_cls_label_flat = tf.reshape(rpn_cls_label, (-1))
        rpn_cls_flat = tf.reshape(rpn_cls, (-1))
        fg_mask = (rpn_cls_label_flat > 0)

        # focal loss
        rpn_cls_target = tf.cast((rpn_cls_label_flat > 0), tf.int32)
        pos = tf.cast((rpn_cls_label_flat > 0), tf.float32)
        neg = tf.cast((rpn_cls_label_flat == 0), tf.float32)
        cls_weights = pos + neg
        pos_normalizer = tf.reduce_sum(pos)
        cls_weights = cls_weights / tf.maximum(pos_normalizer, 1.0)
        rpn_loss_cls = self.loss_cls(rpn_cls_flat,
                                     rpn_cls_target,
                                     cls_weights,
                                     avg_factor=1.0)

        # RPN regression loss
        point_num = rpn_reg.shape[0] * rpn_reg.shape[1]
        fg_sum = tf.reduce_sum(tf.cast(fg_mask, tf.int64)).numpy()
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                get_reg_loss(tf.reshape(rpn_reg, (point_num, -1))[fg_mask],
                                        tf.reshape(rpn_reg_label, (point_num, 7))[fg_mask],
                                        loc_scope=self.proposal_layer.loc_scope,
                                        loc_bin_size=self.proposal_layer.loc_bin_size,
                                        num_head_bin=self.proposal_layer.num_head_bin,
                                        anchor_size=self.proposal_layer.mean_size,
                                        get_xz_fine=self.proposal_layer.loc_xz_fine,
                                        get_y_by_bin=False,
                                        get_ry_fine=False)

            loss_size = 3 * loss_size
            rpn_loss_reg = loss_loc + loss_angle + loss_size
        else:
            rpn_loss_reg = tf.reduce_mean(rpn_reg * 0)

        return {
            "cls": rpn_loss_cls * self.loss_weight[0],
            "reg": rpn_loss_reg * self.loss_weight[1]
        }


class RCNN(tf.keras.layers.Layer):

    def __init__(
            self,
            num_classes,
            in_channels=128,
            SA_config={
                "npoints": [128, 32, -1],
                "radius": [0.2, 0.4, 100],
                "nsample": [64, 64, 64],
                "mlps": [[128, 128, 128], [128, 128, 256], [256, 256, 512]]
            },
            cls_out_ch=[256, 256],
            reg_out_ch=[256, 256],
            db_ratio=0.5,
            use_xyz=True,
            xyz_up_layer=[128, 128],
            head={},
            target_head={},
            loss={}):

        super().__init__()
        self.rcnn_input_channel = 5

        self.pool_extra_width = target_head.get("pool_extra_width", 1.0)
        self.num_points = target_head.get("num_points", 512)

        self.proposal_layer = ProposalLayer(**head)

        self.SA_modules = []
        for i in range(len(SA_config["npoints"])):
            mlps = [in_channels] + SA_config["mlps"][i]
            npoint = SA_config["npoints"][
                i] if SA_config["npoints"][i] != -1 else None
            self.SA_modules.append(
                PointnetSAModule(npoint=npoint,
                                 radius=SA_config["radius"][i],
                                 nsample=SA_config["nsample"][i],
                                 mlp=mlps,
                                 use_xyz=use_xyz,
                                 use_bias=True))
            in_channels = mlps[-1]

        self.xyz_up_layer = gen_CNN([self.rcnn_input_channel] + xyz_up_layer,
                                    conv=tf.keras.layers.Conv2D)
        c_out = xyz_up_layer[-1]
        self.merge_down_layer = gen_CNN([c_out * 2, c_out],
                                        conv=tf.keras.layers.Conv2D)

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes

        layers = []
        for i in range(len(cls_out_ch)):
            layers.extend([
                tf.keras.layers.Conv1D(
                    cls_out_ch[i],
                    1,
                    use_bias=True,
                    data_format="channels_first",
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    bias_initializer=tf.keras.initializers.Constant(0.0)),
                tf.keras.layers.ReLU()
            ])
        layers.append(
            tf.keras.layers.Conv1D(
                cls_channel,
                1,
                use_bias=True,
                data_format="channels_first",
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                bias_initializer=tf.keras.initializers.Constant(0.0)))

        self.cls_blocks = tf.keras.Sequential(layers)

        self.loss_cls = tf.keras.losses.BinaryCrossentropy()

        # regression branch
        per_loc_bin_num = int(self.proposal_layer.loc_scope /
                              self.proposal_layer.loc_bin_size) * 2
        loc_y_bin_num = int(self.proposal_layer.loc_y_scope /
                            self.proposal_layer.loc_y_bin_size) * 2
        reg_channel = per_loc_bin_num * 4 + self.proposal_layer.num_head_bin * 2 + 3
        reg_channel += (1 if not self.proposal_layer.get_y_by_bin else
                        loc_y_bin_num * 2)

        layers = []
        for i in range(len(reg_out_ch)):
            layers.extend([
                tf.keras.layers.Conv1D(
                    reg_out_ch[i],
                    1,
                    use_bias=True,
                    data_format="channels_first",
                    kernel_initializer=tf.keras.initializers.GlorotNormal(),
                    bias_initializer=tf.keras.initializers.Constant(0.0)),
                tf.keras.layers.ReLU()
            ])
        layers.append(
            tf.keras.layers.Conv1D(
                reg_channel,
                1,
                use_bias=True,
                data_format="channels_first",
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.001),
                bias_initializer=tf.keras.initializers.Constant(0.0)))

        self.reg_blocks = tf.keras.Sequential(layers)

        self.proposal_target_layer = ProposalTargetLayer(**target_head)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3]
        features = (tf.transpose(pc[..., 3:],
                                 (0, 2, 1)) if pc.shape[-1] > 3 else None)

        return xyz, features

    def call(self,
             roi_boxes3d,
             gt_boxes3d,
             rpn_xyz,
             rpn_features,
             seg_mask,
             pts_depth,
             training=True):
        pts_extra_input_list = [tf.expand_dims(seg_mask, axis=2)]
        pts_extra_input_list.append(
            tf.expand_dims(pts_depth / 70.0 - 0.5, axis=2))
        pts_extra_input = tf.concat(pts_extra_input_list, axis=2)
        pts_feature = tf.concat((pts_extra_input, rpn_features), axis=2)

        if gt_boxes3d is not None:
            target = self.proposal_target_layer(
                [roi_boxes3d, gt_boxes3d, rpn_xyz, pts_feature])
            for k in target:
                target[k] = tf.stop_gradient(target[k])
            pts_input = tf.concat(
                (target['sampled_pts'], target['pts_feature']), axis=2)
            target['pts_input'] = pts_input
        else:
            pooled_features, pooled_empty_flag = roipool3d_utils.roipool3d_gpu(
                rpn_xyz,
                pts_feature,
                roi_boxes3d,
                self.pool_extra_width,
                sampled_pt_num=self.num_points)

            # canonical transformation
            batch_size = roi_boxes3d.shape[0]
            roi_center = roi_boxes3d[:, :, 0:3]
            poss = []
            for k in range(batch_size):
                pos = pooled_features[k, :, :, :3] - tf.expand_dims(
                    roi_center[k], axis=1)
                pos = rotate_pc_along_y_tf(pos, roi_boxes3d[k, :, 6])
                poss.append(pos)
            pooled_features = tf.concat(
                [tf.stack(poss), pooled_features[:, :, :, 3:]], axis=3)

            pts_input = tf.reshape(
                pooled_features,
                (-1, pooled_features.shape[2], pooled_features.shape[3]))

        xyz, features = self._break_up_pc(pts_input)

        xyz_input = tf.expand_dims(tf.transpose(
            pts_input[..., 0:self.rcnn_input_channel], (0, 2, 1)),
                                   axis=3)
        xyz_feature = self.xyz_up_layer(xyz_input, training=training)

        rpn_feature = tf.expand_dims(tf.transpose(
            pts_input[..., self.rcnn_input_channel:], (0, 2, 1)),
                                     axis=3)

        merged_feature = tf.concat((xyz_feature, rpn_feature), axis=1)
        merged_feature = self.merge_down_layer(merged_feature,
                                               training=training)
        l_xyz, l_features = [xyz], [tf.squeeze(merged_feature, axis=3)]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i],
                                                     l_features[i],
                                                     training=training)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        rcnn_cls = tf.squeeze(tf.transpose(
            self.cls_blocks(l_features[-1], training=training), (0, 2, 1)),
                              axis=1)  # (B, 1 or 2)
        rcnn_reg = tf.squeeze(tf.transpose(
            self.reg_blocks(l_features[-1], training=training), (0, 2, 1)),
                              axis=1)  # (B, C)

        ret_dict = {'rois': roi_boxes3d, 'cls': rcnn_cls, 'reg': rcnn_reg}

        if gt_boxes3d is not None:
            ret_dict.update(target)
        return ret_dict

    def loss(self, results, inputs):
        rcnn_cls = results['cls']
        rcnn_reg = results['reg']

        cls_label = tf.cast(results['cls_label'], tf.float32)
        reg_valid_mask = results['reg_valid_mask']
        gt_boxes3d_ct = results['gt_of_rois']
        pts_input = results['pts_input']

        cls_label_flat = tf.reshape(cls_label, (-1))

        # binary cross entropy
        rcnn_cls_flat = tf.reshape(rcnn_cls, (-1))
        batch_loss_cls = tf.keras.losses.BinaryCrossentropy(reduction="none")(
            tf.sigmoid(rcnn_cls_flat), cls_label)
        cls_valid_mask = tf.cast((cls_label_flat >= 0), tf.float32)
        rcnn_loss_cls = tf.reduce_sum(
            batch_loss_cls * cls_valid_mask) / tf.maximum(
                tf.reduce_sum(cls_valid_mask), 1.0)

        # rcnn regression loss
        batch_size = pts_input.shape[0]
        fg_mask = (reg_valid_mask > 0)
        fg_sum = tf.reduce_sum(tf.cast(fg_mask, tf.int64)).numpy()
        if fg_sum != 0:
            anchor_size = self.proposal_layer.mean_size

            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                get_reg_loss(tf.reshape(rcnn_reg, (batch_size, -1))[fg_mask],
                                        tf.reshape(gt_boxes3d_ct, (batch_size, 7))[fg_mask],
                                        loc_scope=self.proposal_layer.loc_scope,
                                        loc_bin_size=self.proposal_layer.loc_bin_size,
                                        num_head_bin=self.proposal_layer.num_head_bin,
                                        anchor_size=anchor_size,
                                        get_xz_fine=True, get_y_by_bin=self.proposal_layer.get_y_by_bin,
                                        loc_y_scope=self.proposal_layer.loc_y_scope, loc_y_bin_size=self.proposal_layer.loc_y_bin_size,
                                        get_ry_fine=True)

            loss_size = 3 * loss_size  # consistent with old codes
            rcnn_loss_reg = loss_loc + loss_angle + loss_size
        else:
            #  Regression loss is zero when no point is classified as foreground.
            rcnn_loss_reg = tf.reduce_mean(rcnn_reg * 0)

        return {"cls": rcnn_loss_cls, "reg": rcnn_loss_reg}


def rotate_pc_along_y(pc, rot_angle):
    """
    Args:
        pc: (N, 3+C), (N, 3) is in the rectified camera coordinate.
        rot_angle: rad scalar

    Returns:
        pc: updated pc with XYZ rotated.
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


class ProposalLayer(tf.keras.layers.Layer):

    def __init__(self,
                 nms_pre=9000,
                 nms_post=512,
                 nms_thres=0.85,
                 nms_post_val=None,
                 nms_thres_val=None,
                 mean_size=[1.0],
                 loc_xz_fine=True,
                 loc_scope=3.0,
                 loc_bin_size=0.5,
                 num_head_bin=12,
                 get_y_by_bin=False,
                 get_ry_fine=False,
                 loc_y_scope=0.5,
                 loc_y_bin_size=0.25,
                 post_process=True):
        super().__init__()
        self.nms_pre = nms_pre
        self.nms_post = nms_post
        self.nms_thres = nms_thres
        self.nms_post_val = nms_post_val
        self.nms_thres_val = nms_thres_val
        self.mean_size = tf.constant(mean_size)
        self.loc_scope = loc_scope
        self.loc_bin_size = loc_bin_size
        self.num_head_bin = num_head_bin
        self.loc_xz_fine = loc_xz_fine
        self.get_y_by_bin = get_y_by_bin
        self.get_ry_fine = get_ry_fine
        self.loc_y_scope = loc_y_scope
        self.loc_y_bin_size = loc_y_bin_size
        self.post_process = post_process

    def call(self, rpn_scores, rpn_reg, xyz, training=True):
        batch_size = xyz.shape[0]
        proposals = decode_bbox_target(
            tf.reshape(xyz, (-1, xyz.shape[-1])),
            tf.reshape(rpn_reg, (-1, rpn_reg.shape[-1])),
            anchor_size=self.mean_size,
            loc_scope=self.loc_scope,
            loc_bin_size=self.loc_bin_size,
            num_head_bin=self.num_head_bin,
            get_xz_fine=self.loc_xz_fine,
            get_y_by_bin=self.get_y_by_bin,
            get_ry_fine=self.get_ry_fine,
            loc_y_scope=self.loc_y_scope,
            loc_y_bin_size=self.loc_y_bin_size)  # (N, 7)

        proposals = tf.reshape(proposals, (batch_size, -1, 7))

        nms_post = self.nms_post
        nms_thres = self.nms_thres
        if not training:
            if self.nms_post_val is not None:
                nms_post = self.nms_post_val
            if self.nms_thres_val is not None:
                nms_thres = self.nms_thres_val

        if self.post_process:
            proposals = tf.concat([
                proposals[..., :1], proposals[..., 1:2] +
                proposals[..., 3:4] / 2, proposals[..., 2:]
            ],
                                  axis=-1)  # set y as the center of bottom
            scores = rpn_scores
            sorted_idxs = tf.argsort(scores, axis=1, direction="DESCENDING")

            batch_size = scores.shape[0]
            ret_bbox3d = []
            ret_scores = []
            for k in range(batch_size):
                scores_single = scores[k]
                proposals_single = proposals[k]
                order_single = sorted_idxs[k]

                scores_single, proposals_single = self.distance_based_proposal(
                    scores_single, proposals_single, order_single, training)

                proposals_tot = proposals_single.shape[0]

                ret_bbox3d.append(
                    tf.concat([
                        proposals_single,
                        tf.zeros((nms_post - proposals_tot, 7))
                    ],
                              axis=0))
                ret_scores.append(
                    tf.concat(
                        [scores_single,
                         tf.zeros((nms_post - proposals_tot,))],
                        axis=0))
            ret_bbox3d = tf.stack(ret_bbox3d)
            ret_scores = tf.stack(ret_scores)
        else:
            batch_size = rpn_scores.shape[0]
            ret_bbox3d = []
            ret_scores = []
            for k in range(batch_size):
                bev = xywhr_to_xyxyr(
                    tf.stack([proposals[k, :, i] for i in [0, 2, 3, 5, 6]],
                             axis=-1))
                keep_idx = nms(bev, rpn_scores[k, :, 0], nms_thres)

                ret_bbox3d.append(tf.gather(proposals[k], keep_idx))
                ret_scores.append(tf.gather(rpn_scores[k], keep_idx))

        return ret_bbox3d, ret_scores

    def distance_based_proposal(self, scores, proposals, order, training=True):
        """Propose ROIs in two area based on the distance.

        Args:
            scores: (N)
            proposals: (N, 7)
            order: (N)
            training (bool): Whether we are training?
        """
        nms_post = self.nms_post
        nms_thres = self.nms_thres
        if not training:
            if self.nms_post_val is not None:
                nms_post = self.nms_post_val
            if self.nms_thres_val is not None:
                nms_thres = self.nms_thres_val

        nms_range_list = [0, 40.0, 80.0]
        pre_top_n_list = [
            0,
            int(self.nms_pre * 0.7), self.nms_pre - int(self.nms_pre * 0.7)
        ]
        post_top_n_list = [
            0, int(nms_post * 0.7), nms_post - int(nms_post * 0.7)
        ]

        scores_single_list, proposals_single_list = [], []

        # sort by score
        scores_ordered = tf.gather(scores, order)
        proposals_ordered = tf.gather(proposals, order)

        dist = proposals_ordered[:, 2]
        first_mask = (dist > nms_range_list[0]) & (dist <= nms_range_list[1])
        for i in range(1, len(nms_range_list)):
            # get proposal distance mask
            dist_mask = ((dist > nms_range_list[i - 1]) &
                         (dist <= nms_range_list[i]))

            if tf.reduce_any(dist_mask):
                # this area has points
                # reduce by mask
                cur_scores = scores_ordered[dist_mask]
                cur_proposals = proposals_ordered[dist_mask]

                # fetch pre nms top K
                cur_scores = cur_scores[:pre_top_n_list[i]]
                cur_proposals = cur_proposals[:pre_top_n_list[i]]
            else:
                assert i == 2, '%d' % i
                # this area doesn't have any points, so use rois of first area
                cur_scores = scores_ordered[first_mask]
                cur_proposals = proposals_ordered[first_mask]

                # fetch top K of first area
                cur_scores = cur_scores[pre_top_n_list[i -
                                                       1]:][:pre_top_n_list[i]]
                cur_proposals = cur_proposals[
                    pre_top_n_list[i - 1]:][:pre_top_n_list[i]]

            # oriented nms
            bev = xywhr_to_xyxyr(
                tf.gather(cur_proposals, [0, 2, 3, 5, 6], axis=1))
            keep_idx = nms(bev, cur_scores, nms_thres)

            # Fetch post nms top k
            keep_idx = keep_idx[:post_top_n_list[i]]

            scores_single_list.append(tf.gather(cur_scores, keep_idx))
            proposals_single_list.append(tf.gather(cur_proposals, keep_idx))

        scores_single = tf.concat(scores_single_list, axis=0)
        proposals_single = tf.concat(proposals_single_list, axis=0)
        return scores_single, proposals_single


def decode_bbox_target(roi_box3d,
                       pred_reg,
                       loc_scope,
                       loc_bin_size,
                       num_head_bin,
                       anchor_size,
                       get_xz_fine=True,
                       get_y_by_bin=False,
                       loc_y_scope=0.5,
                       loc_y_bin_size=0.25,
                       get_ry_fine=False):
    """
    Args:
        roi_box3d: (N, 7)
        pred_reg: (N, C)
        loc_scope:
        loc_bin_size:
        num_head_bin:
        anchor_size:
        get_xz_fine:
        get_y_by_bin:
        loc_y_scope:
        loc_y_bin_size:
        get_ry_fine:
    """
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
    loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

    # recover xz localization
    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r

    x_bin = tf.argmax(pred_reg[:, x_bin_l:x_bin_r], axis=1)
    z_bin = tf.argmax(pred_reg[:, z_bin_l:z_bin_r], axis=1)

    pos_x = tf.cast(x_bin,
                    tf.float32) * loc_bin_size + loc_bin_size / 2 - loc_scope
    pos_z = tf.cast(z_bin,
                    tf.float32) * loc_bin_size + loc_bin_size / 2 - loc_scope

    if get_xz_fine:
        x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
        z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
        start_offset = z_res_r

        x_res_norm = tf.gather(pred_reg[:, x_res_l:x_res_r],
                               x_bin,
                               batch_dims=1)
        z_res_norm = tf.gather(pred_reg[:, z_res_l:z_res_r],
                               z_bin,
                               batch_dims=1)
        x_res = x_res_norm * loc_bin_size
        z_res = z_res_norm * loc_bin_size

        pos_x += x_res
        pos_z += z_res

    # recover y localization
    if get_y_by_bin:
        y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
        y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
        start_offset = y_res_r

        y_bin = tf.argmax(pred_reg[:, y_bin_l:y_bin_r], axis=1)
        y_res_norm = tf.gather(pred_reg[:, y_res_l:y_res_r],
                               y_bin,
                               batch_dims=1)
        y_res = y_res_norm * loc_y_bin_size
        pos_y = tf.cast(
            y_bin, tf.float32
        ) * loc_y_bin_size + loc_y_bin_size / 2 - loc_y_scope + y_res
        pos_y = pos_y + roi_box3d[:, 1]
    else:
        y_offset_l, y_offset_r = start_offset, start_offset + 1
        start_offset = y_offset_r

        pos_y = roi_box3d[:, 1] + pred_reg[:, y_offset_l]

    # recover ry rotation
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_bin = tf.argmax(pred_reg[:, ry_bin_l:ry_bin_r], axis=1)
    ry_res_norm = tf.gather(pred_reg[:, ry_res_l:ry_res_r],
                            ry_bin,
                            batch_dims=1)
    if get_ry_fine:
        # divide pi/2 into several bins
        angle_per_class = (np.pi / 2) / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)
        ry = (tf.cast(ry_bin, tf.float32) * angle_per_class +
              angle_per_class / 2) + ry_res - np.pi / 4
    else:
        angle_per_class = (2 * np.pi) / num_head_bin
        ry_res = ry_res_norm * (angle_per_class / 2)

        # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
        ry = (tf.cast(ry_bin, tf.float32) * angle_per_class + ry_res) % (2 *
                                                                         np.pi)
        ry = tf.where(ry > np.pi, ry - 2 * np.pi, ry)

    # recover size
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert size_res_r == pred_reg.shape[1]

    size_res_norm = pred_reg[:, size_res_l:size_res_r]
    hwl = size_res_norm * anchor_size + anchor_size

    # shift to original coords
    roi_center = roi_box3d[:, 0:3]
    shift_ret_box3d = tf.concat(
        (tf.reshape(pos_x, (-1, 1)), tf.reshape(
            pos_y, (-1, 1)), tf.reshape(pos_z,
                                        (-1, 1)), hwl, tf.reshape(ry, (-1, 1))),
        axis=1)
    ret_box3d = shift_ret_box3d
    if roi_box3d.shape[1] == 7:
        roi_ry = roi_box3d[:, 6:7]
        ret_box3d = rotate_pc_along_y_tf(shift_ret_box3d, -roi_ry)
        ret_box3d = tf.concat([ret_box3d[:, :6], ret_box3d[:, 6:7] + roi_ry],
                              axis=1)
    ret_box3d = tf.concat([
        ret_box3d[:, :1] + roi_center[:, :1], ret_box3d[:, 1:2],
        ret_box3d[:, 2:3] + roi_center[:, 2:3], ret_box3d[:, 3:]
    ],
                          axis=1)

    return ret_box3d


def rotate_pc_along_y_tf(pc, rot_angle):
    """
    :param pc: (N, 3 + C)
    :param rot_angle: (N)
    :return:
    """
    cosa = tf.reshape(tf.cos(rot_angle), (-1, 1))  # (N, 1)
    sina = tf.reshape(tf.sin(rot_angle), (-1, 1))  # (N, 1)

    raw_1 = tf.concat([cosa, -sina], axis=1)  # (N, 2)
    raw_2 = tf.concat([sina, cosa], axis=1)  # (N, 2)
    R = tf.concat(
        (tf.expand_dims(raw_1, axis=1), tf.expand_dims(raw_2, axis=1)),
        axis=1)  # (N, 2, 2)

    pc_temp = tf.reshape(tf.stack([pc[..., 0], pc[..., 2]], axis=-1),
                         ((pc.shape[0], -1, 2)))  # (N, 512, 2)
    pc_temp = tf.matmul(pc_temp, tf.transpose(R, (0, 2, 1)))
    pc_temp = tf.reshape(pc_temp, (pc.shape[:-1] + (2,)))  # (N, 512, 2)

    pc = tf.concat(
        [pc_temp[..., :1], pc[..., 1:2], pc_temp[..., 1:2], pc[..., 3:]],
        axis=-1)

    return pc


class ProposalTargetLayer(tf.keras.layers.Layer):

    def __init__(self,
                 pool_extra_width=1.0,
                 num_points=512,
                 reg_fg_thresh=0.55,
                 cls_fg_thresh=0.6,
                 cls_bg_thresh=0.45,
                 cls_bg_thresh_lo=0.05,
                 fg_ratio=0.5,
                 roi_per_image=64,
                 aug_rot_range=18,
                 hard_bg_ratio=0.8,
                 roi_fg_aug_times=10):
        super().__init__()
        self.pool_extra_width = pool_extra_width
        self.num_points = num_points
        self.reg_fg_thresh = reg_fg_thresh
        self.cls_fg_thresh = cls_fg_thresh
        self.cls_bg_thresh = cls_bg_thresh
        self.cls_bg_thresh_lo = cls_bg_thresh_lo
        self.fg_ratio = fg_ratio
        self.roi_per_image = roi_per_image
        self.aug_rot_range = aug_rot_range
        self.hard_bg_ratio = hard_bg_ratio
        self.roi_fg_aug_times = roi_fg_aug_times

    def call(self, x):
        roi_boxes3d, gt_boxes3d, rpn_xyz, pts_feature = x
        batch_rois, batch_gt_of_rois, batch_roi_iou = self.sample_rois_for_rcnn(
            roi_boxes3d, gt_boxes3d)

        # point cloud pooling
        pooled_features, pooled_empty_flag = \
            roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, self.pool_extra_width,
                                          sampled_pt_num=self.num_points)

        sampled_pts, sampled_features = pooled_features[:, :, :, 0:
                                                        3], pooled_features[:, :, :,
                                                                            3:]

        # data augmentation
        sampled_pts, batch_rois, batch_gt_of_rois = \
            self.data_augmentation(sampled_pts, batch_rois, batch_gt_of_rois)

        # canonical transformation
        batch_size = batch_rois.shape[0]
        roi_ry = batch_rois[:, :, 6:7] % (2 * np.pi)
        roi_center = batch_rois[:, :, 0:3]
        sampled_pts = sampled_pts - tf.expand_dims(roi_center,
                                                   axis=2)  # (B, M, 512, 3)
        batch_gt_of_rois = tf.concat([
            batch_gt_of_rois[:, :, :3] - roi_center,
            batch_gt_of_rois[:, :, 3:6], batch_gt_of_rois[:, :, 6:] - roi_ry
        ],
                                     axis=2)

        sampled_pts = tf.unstack(sampled_pts)
        batch_gt_of_rois = tf.unstack(batch_gt_of_rois)
        for k in range(batch_size):
            sampled_pts[k] = rotate_pc_along_y_tf(sampled_pts[k],
                                                  batch_rois[k, :, 6])
            batch_gt_of_rois[k] = tf.squeeze(rotate_pc_along_y_tf(
                tf.expand_dims(batch_gt_of_rois[k], axis=1), roi_ry[k]),
                                             axis=1)
        sampled_pts = tf.stack(sampled_pts)
        batch_gt_of_rois = tf.stack(batch_gt_of_rois)

        # regression valid mask
        valid_mask = (pooled_empty_flag == 0)
        reg_valid_mask = tf.cast(
            ((batch_roi_iou > self.reg_fg_thresh) & valid_mask), tf.int64)

        # classification label
        batch_cls_label = tf.cast((batch_roi_iou > self.cls_fg_thresh),
                                  tf.int64)
        invalid_mask = (batch_roi_iou > self.cls_bg_thresh) & (
            batch_roi_iou < self.cls_fg_thresh)
        batch_cls_label = tf.where(
            tf.reduce_any([tf.logical_not(valid_mask), invalid_mask], axis=0),
            -1, batch_cls_label)

        output_dict = {
            'sampled_pts':
                tf.reshape(sampled_pts, (-1, self.num_points, 3)),
            'pts_feature':
                tf.reshape(sampled_features,
                           (-1, self.num_points, sampled_features.shape[3])),
            'cls_label':
                tf.reshape(batch_cls_label, (-1)),
            'reg_valid_mask':
                tf.reshape(reg_valid_mask, (-1)),
            'gt_of_rois':
                tf.reshape(batch_gt_of_rois, (-1, 7)),
            'gt_iou':
                tf.reshape(batch_roi_iou, (-1)),
            'roi_boxes3d':
                tf.reshape(batch_rois, (-1, 7))
        }

        return output_dict

    def sample_rois_for_rcnn(self, roi_boxes3d, gt_boxes3d):
        """

        Args:
            roi_boxes3d: (B, M, 7)
            gt_boxes3d: (B, N, 8) [x, y, z, h, w, l, ry, cls]

        Returns:
            batch_rois: (B, N, 7)
            batch_gt_of_rois: (B, N, 8)
            batch_roi_iou: (B, N)
        """
        batch_size = roi_boxes3d.shape[0]

        fg_rois_per_image = int(np.round(self.fg_ratio * self.roi_per_image))

        batch_rois, batch_gt_of_rois, batch_roi_iou = [], [], []
        for idx in range(batch_size):
            cur_roi, cur_gt = roi_boxes3d[idx], gt_boxes3d[idx]

            k = cur_gt.__len__() - 1
            while k >= 0 and tf.reduce_sum(cur_gt[k]) == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]

            if cur_gt.__len__() == 0:
                cur_gt = tf.zeros((1, 7), tf.float32)

            # include gt boxes in the candidate rois
            iou3d = iou_3d(cur_roi.numpy()[:, [0, 1, 2, 5, 3, 4, 6]],
                           cur_gt[:,
                                  0:7].numpy()[:,
                                               [0, 1, 2, 5, 3, 4, 6]])  # (M, N)
            iou3d = tf.constant(iou3d)

            gt_assignment = tf.argmax(iou3d, axis=1)
            max_overlaps = tf.gather(iou3d, gt_assignment, batch_dims=1)

            # sample fg, easy_bg, hard_bg
            fg_thresh = min(self.reg_fg_thresh, self.cls_fg_thresh)
            fg_inds = tf.reshape(tf.where((max_overlaps >= fg_thresh)), (-1))

            # TODO: this will mix the fg and bg when CLS_BG_THRESH_LO < iou < CLS_BG_THRESH
            # fg_inds = tf.concat((fg_inds, roi_assignment), axis=0)  # consider the roi which has max_iou with gt as fg

            easy_bg_inds = tf.reshape(
                tf.where((max_overlaps < self.cls_bg_thresh_lo)), (-1))
            hard_bg_inds = tf.reshape(
                tf.where((max_overlaps < self.cls_bg_thresh) &
                         (max_overlaps >= self.cls_bg_thresh_lo)), (-1))

            fg_num_rois = len(fg_inds.shape)
            bg_num_rois = len(hard_bg_inds.shape) + len(easy_bg_inds.shape)

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                rand_num = tf.constant(np.random.permutation(fg_num_rois),
                                       dtype=tf.int64)
                fg_inds = tf.gather(fg_inds, rand_num[:fg_rois_per_this_image])

                # sampling bg
                bg_rois_per_this_image = self.roi_per_image - fg_rois_per_this_image
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds,
                                              bg_rois_per_this_image)

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                rand_num = np.floor(
                    np.random.rand(self.roi_per_image) * fg_num_rois)
                rand_num = tf.constant(rand_num, dtype=tf.int64)
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = self.roi_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                bg_rois_per_this_image = self.roi_per_image
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds,
                                              bg_rois_per_this_image)

                fg_rois_per_this_image = 0
            else:
                import pdb
                pdb.set_trace()
                raise NotImplementedError

            # augment the rois by noise
            roi_list, roi_iou_list, roi_gt_list = [], [], []
            if fg_rois_per_this_image > 0:
                fg_rois_src = tf.gather(cur_roi, fg_inds)
                gt_of_fg_rois = tf.gather(cur_gt,
                                          tf.gather(gt_assignment, fg_inds))
                iou3d_src = tf.gather(max_overlaps, fg_inds)
                fg_rois, fg_iou3d = self.aug_roi_by_noise_torch(
                    fg_rois_src,
                    gt_of_fg_rois,
                    iou3d_src,
                    aug_times=self.roi_fg_aug_times)
                roi_list.append(fg_rois)
                roi_iou_list.append(fg_iou3d)
                roi_gt_list.append(gt_of_fg_rois)

            if bg_rois_per_this_image > 0:
                bg_rois_src = tf.gather(cur_roi, bg_inds)
                gt_of_bg_rois = tf.gather(cur_gt,
                                          tf.gather(gt_assignment, bg_inds))
                iou3d_src = tf.gather(max_overlaps, bg_inds)
                aug_times = 1 if self.roi_fg_aug_times > 0 else 0
                bg_rois, bg_iou3d = self.aug_roi_by_noise_torch(
                    bg_rois_src, gt_of_bg_rois, iou3d_src, aug_times=aug_times)
                roi_list.append(bg_rois)
                roi_iou_list.append(bg_iou3d)
                roi_gt_list.append(gt_of_bg_rois)

            rois = tf.concat(roi_list, axis=0)
            iou_of_rois = tf.concat(roi_iou_list, axis=0)
            gt_of_rois = tf.concat(roi_gt_list, axis=0)

            batch_rois.append(rois)
            batch_gt_of_rois.append(gt_of_rois)
            batch_roi_iou.append(iou_of_rois)

        return tf.stack(batch_rois), tf.stack(batch_gt_of_rois), tf.stack(
            batch_roi_iou)

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds,
                       bg_rois_per_this_image):
        if len(hard_bg_inds.shape) > 0 and len(easy_bg_inds.shape) > 0:
            hard_bg_rois_num = int(bg_rois_per_this_image * self.hard_bg_ratio)
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = tf.constant(np.random.randint(low=0,
                                                     high=len(
                                                         hard_bg_inds.shape),
                                                     size=(hard_bg_rois_num,)),
                                   dtype=tf.int64)
            hard_bg_inds = tf.gather(hard_bg_inds, rand_idx)

            # sampling easy bg
            rand_idx = tf.constant(np.random.randint(low=0,
                                                     high=len(
                                                         easy_bg_inds.shape),
                                                     size=(easy_bg_rois_num,)),
                                   dtype=tf.int64)
            easy_bg_inds = tf.gather(easy_bg_inds, rand_idx)

            bg_inds = tf.concat([hard_bg_inds, easy_bg_inds], axis=0)
        elif len(hard_bg_inds.shape) > 0 and len(easy_bg_inds.shape) == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = tf.constant(np.random.randint(low=0,
                                                     high=len(
                                                         hard_bg_inds.shape),
                                                     size=(hard_bg_rois_num,)),
                                   dtype=tf.int64)
            bg_inds = tf.gather(hard_bg_inds, rand_idx)
        elif len(hard_bg_inds.shape) == 0 and len(easy_bg_inds.shape) > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = tf.constant(np.random.randint(low=0,
                                                     high=len(
                                                         easy_bg_inds.shape),
                                                     size=(easy_bg_rois_num,)),
                                   dtype=tf.int64)
            bg_inds = tf.gather(easy_bg_inds, rand_idx)
        else:
            raise NotImplementedError

        return bg_inds

    def aug_roi_by_noise_torch(self,
                               roi_boxes3d,
                               gt_boxes3d,
                               iou3d_src,
                               aug_times=10):
        pos_thresh = min(self.reg_fg_thresh, self.cls_fg_thresh)

        aug_boxes = []
        iou_of_rois = []
        for k in range(roi_boxes3d.shape[0]):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]

            gt_box3d = tf.reshape(gt_boxes3d[k], (1, 7))
            aug_box3d = roi_box3d
            keep = True
            while temp_iou < pos_thresh and cnt < aug_times:
                if np.random.rand() < 0.2:
                    aug_box3d = roi_box3d  # p=0.2 to keep the original roi box
                    keep = True
                else:
                    aug_box3d = self.random_aug_box3d(roi_box3d)
                    keep = False
                aug_box3d = tf.reshape(aug_box3d, ((1, 7)))

                iou3d = iou_3d(aug_box3d.numpy()[:, [0, 1, 2, 5, 3, 4, 6]],
                               gt_box3d.numpy()[:, [0, 1, 2, 5, 3, 4, 6]])
                iou3d = tf.constant(iou3d)
                temp_iou = iou3d[0][0]
                cnt += 1
            aug_boxes.append(tf.reshape(aug_box3d, (-1)))
            if cnt == 0 or keep:
                iou_of_rois.append(iou3d_src[k])
            else:
                iou_of_rois.append(temp_iou)
        return tf.stack(aug_boxes), tf.stack(iou_of_rois)

    @staticmethod
    def random_aug_box3d(box3d):
        """
        Random shift, scale, orientation.

        Args:
            box3d: (7) [x, y, z, h, w, l, ry]
        """
        # pos_range, hwl_range, angle_range, mean_iou
        range_config = [[0.2, 0.1, np.pi / 12,
                         0.7], [0.3, 0.15, np.pi / 12, 0.6],
                        [0.5, 0.15, np.pi / 9,
                         0.5], [0.8, 0.15, np.pi / 6, 0.3],
                        [1.0, 0.15, np.pi / 3, 0.2]]
        idx = tf.constant(np.random.randint(low=0,
                                            high=len(range_config),
                                            size=(1,))[0],
                          dtype=tf.int64)

        pos_shift = ((tf.random.uniform(
            (3,)) - 0.5) / 0.5) * range_config[idx][0]
        hwl_scale = ((tf.random.uniform(
            (3,)) - 0.5) / 0.5) * range_config[idx][1] + 1.0
        angle_rot = ((tf.random.uniform(
            (1,)) - 0.5) / 0.5) * range_config[idx][2]

        aug_box3d = tf.concat([
            box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale,
            box3d[6:7] + angle_rot
        ],
                              axis=0)
        return aug_box3d

    def data_augmentation(self, pts, rois, gt_of_rois):
        """
        Args:
            pts: (B, M, 512, 3)
            rois: (B, M. 7)
            gt_of_rois: (B, M, 7)
        """
        batch_size, boxes_num = pts.shape[0], pts.shape[1]

        # rotation augmentation
        angles = (tf.random.uniform(
            (batch_size, boxes_num)) - 0.5 / 0.5) * (np.pi / self.aug_rot_range)

        # calculate gt alpha from gt_of_rois
        temp_x, temp_z, temp_ry = gt_of_rois[:, :,
                                             0], gt_of_rois[:, :,
                                                            2], gt_of_rois[:, :,
                                                                           6]
        temp_beta = tf.atan2(temp_z, temp_x)
        gt_alpha = -tf.sign(
            temp_beta) * np.pi / 2 + temp_beta + temp_ry  # (B, M)

        temp_x, temp_z, temp_ry = rois[:, :, 0], rois[:, :, 2], rois[:, :, 6]
        temp_beta = tf.atan2(temp_z, temp_x)
        roi_alpha = -tf.sign(
            temp_beta) * np.pi / 2 + temp_beta + temp_ry  # (B, M)

        pts = tf.unstack(pts)
        gt_of_rois = tf.unstack(gt_of_rois)
        rois = tf.unstack(rois)
        for k in range(batch_size):
            pts[k] = rotate_pc_along_y_tf(pts[k], angles[k])
            gt_of_rois[k] = tf.squeeze(rotate_pc_along_y_tf(
                tf.expand_dims(gt_of_rois[k], axis=1), angles[k]),
                                       axis=1)
            rois[k] = tf.squeeze(rotate_pc_along_y_tf(
                tf.expand_dims(rois[k], axis=1), angles[k]),
                                 axis=1)

        pts = tf.stack(pts)
        gt_of_rois = tf.stack(gt_of_rois)
        rois = tf.stack(rois)

        # calculate the ry after rotation
        temp_x, temp_z = gt_of_rois[:, :, :1], gt_of_rois[:, :, 2:3]
        temp_beta = tf.atan2(temp_z, temp_x)
        gt_of_rois = tf.concat([
            gt_of_rois[:, :, :6],
            tf.sign(temp_beta) * np.pi / 2 + tf.expand_dims(gt_alpha, axis=-1) -
            temp_beta
        ],
                               axis=2)

        temp_x, temp_z = rois[:, :, :1], rois[:, :, 2:3]
        temp_beta = tf.atan2(temp_z, temp_x)
        rois = tf.concat([
            rois[:, :, :6],
            tf.sign(temp_beta) * np.pi / 2 +
            tf.expand_dims(roi_alpha, axis=-1) - temp_beta
        ],
                         axis=2)

        # scaling augmentation
        scales = 1 + ((tf.random.uniform(
            (batch_size, boxes_num)) - 0.5) / 0.5) * 0.05
        pts = pts * tf.expand_dims(tf.expand_dims(scales, axis=2), axis=3)
        gt_of_rois = tf.concat([
            gt_of_rois[:, :, :6] * tf.expand_dims(scales, axis=2),
            gt_of_rois[:, :, 6:]
        ],
                               axis=2)
        rois = tf.concat(
            [rois[:, :, :6] * tf.expand_dims(scales, axis=2), rois[:, :, 6:]],
            axis=2)

        # flip augmentation
        flip_flag = tf.sign(tf.random.uniform((batch_size, boxes_num, 1)) - 0.5)
        pts = tf.concat([
            pts[:, :, :, :1] * tf.expand_dims(flip_flag, axis=3), pts[:, :, :,
                                                                      1:]
        ],
                        axis=3)
        gt_of_rois = tf.concat(
            [gt_of_rois[:, :, :1] * flip_flag, gt_of_rois[:, :, 1:]], axis=2)
        # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
        src_ry = gt_of_rois[:, :, 6:7]
        ry = tf.cast((flip_flag == 1), tf.float32) * src_ry + tf.cast(
            (flip_flag == -1), tf.float32) * (tf.sign(src_ry) * np.pi - src_ry)
        gt_of_rois = tf.concat([gt_of_rois[:, :, :6], ry], axis=2)

        rois = tf.concat([rois[:, :, :1] * flip_flag, rois[:, :, 1:]], axis=2)
        # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
        src_ry = rois[:, :, 6:7]
        ry = tf.cast((flip_flag == 1), tf.float32) * src_ry + tf.cast(
            (flip_flag == -1), tf.float32) * (tf.sign(src_ry) * np.pi - src_ry)
        rois = tf.concat([rois[:, :, :6], ry], axis=2)

        return pts, rois, gt_of_rois
