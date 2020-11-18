#TODO ref orig impl

import torch
from torch import nn
from torch.nn import functional as F
from functools import partial

import numpy as np

from .base_model import BaseModel

from ...utils import MODEL
from ..utils.objdet_helper import Anchor3DRangeGenerator, DeltaXYZWLHRBBoxCoder, xywhr2xyxyr, LiDARInstance3DBoxes, box3d_multiclass_nms, limit_period, multi_apply, MaxIoUAssigner, PseudoSampler, get_direction_target, images_to_levels, bbox2result_kitti
from ..modules.losses.focal_loss import FocalLoss
from ..modules.losses.smooth_L1 import SmoothL1Loss
from ..modules.losses.cross_entropy import CrossEntropyLoss

#from mmdet3d.ops import Voxelization
from .point_pillars_voxelize import PointPillarsVoxelization

class PointPillars(BaseModel):
    def __init__(self, 
            name="PointPillars",
            ckpt_path=None):

        super().__init__(
                name=name,
                ckpt_path=ckpt_path)

        self.backbone = SECOND()
        self.neck = SECONDFPN()
        self.bbox_head = Anchor3DHead()

        self.voxel_layer = PointPillarsVoxelization(
            voxel_size=[0.16, 0.16, 4],
            point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
            max_num_points=32,
            max_voxels=(16000, 40000)
        )
        self.voxel_encoder = PillarFeatureNet()
        self.middle_encoder = PointPillarsScatter()

    def extract_feats(self, points):
        """Extract features from points."""
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        x = self.neck(x)
        return x

    @torch.no_grad()
    #@force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward(self, inputs):
        x = self.extract_feats(inputs)
        outs = self.bbox_head(x)
        return outs

    def get_optimizer(self, cfg_pipeline):
        raise NotImplementedError
    
    def get_loss(self, Loss, results, inputs):
        
        cls_scores, bbox_preds, dir_cls_preds = results
        gt_bboxes, gt_labels = inputs
        losses = self.bbox_head.loss(
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            gt_bboxes,
            gt_labels)

        return losses

    def preprocess(self, data, attr):
        return data

    def transform(self, data, attr):
        #data = data['data']
        points = np.array(data['point'][:, 0:4], dtype=np.float32)       

        min_val = np.array([0.0, -40.0, -3.0])
        max_val = np.array([70.4, 40.0, 1.0])

        points = points[np.where(np.all( 
            np.logical_and(
                points[:,:3] >= min_val, 
                points[:,:3] < max_val), axis=-1))]

        if 'label' not in data.keys() or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = data['label']

        if 'feat' not in data.keys() or data['feat'] is None:
            feat = None
        else:
            feat = np.array(data['feat'], dtype=np.float32)

        data = dict()
        data['point'] = points
        data['feat'] = feat
        data['label'] = labels

        return data

    def inference_begin(self, data):
        self.inference_data = data

    def inference_preprocess(self):
        data = torch.tensor([self.inference_data["point"]], dtype=torch.float32, device=self.device)
        return {
            "data": data
        }

    def inference_end(self, inputs, results):
        bbox_list = self.bbox_head.get_bboxes(*results)
        self.inference_result = [
            dict(
                boxes_3d=bboxes.to('cpu'),
                scores_3d=scores.cpu(),
                labels_3d=labels.cpu()) 
            for bboxes, scores, labels in bbox_list
        ]
        #self.inference_result = bbox2result_kitti(result, self.inference_data, ['Car', 'Pedestrian', 'Cyclist'])
        return True


MODEL._register_module(PointPillars, 'torch')


class SECOND(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
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
            block = [
                nn.Conv2d(
                    in_filters[i],
                    out_channels[i],
                    3,
                    bias=False,
                    stride=layer_strides[i],
                    padding=1),
                nn.BatchNorm2d(out_channels[i], 
                    eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    nn.Conv2d(
                        out_channels[i],
                        out_channels[i],
                        3,
                        bias=False,
                        padding=1))
                block.append(nn.BatchNorm2d(out_channels[i], 
                    eps=1e-3, momentum=0.01))
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)


class SECONDFPN(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[64, 128, 256],
                 out_channels=[128, 128, 128],
                 upsample_strides=[1, 2, 4],
                 use_conv_for_no_stride=False):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = nn.ConvTranspose2d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False)
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride,
                    bias=False)

            deblock = nn.Sequential(upsample_layer,
                                    nn.BatchNorm2d(out_channel, 
                                        eps=1e-3, momentum=0.01),
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        # TODO: check!!!
        #self.init_weights()

    def init_weights(self):
        """Initialize weights of FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    #@auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]


class Anchor3DHead(nn.Module):
    def __init__(self,
                 num_classes=3,
                 in_channels=384,
                 feat_channels=384): # TODO

        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        
        # build anchor generator
        self.anchor_generator = Anchor3DRangeGenerator(
            ranges=[
                [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                [0, -39.68, -0.6, 70.4, 39.68, -0.6],
                [0, -39.68, -1.78, 70.4, 39.68, -1.78],
            ],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False)
        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.anchor_generator.num_base_anchors
        # build box coder
        self.bbox_coder = DeltaXYZWLHRBBoxCoder()
        self.box_code_size = self.bbox_coder.code_size

        self.bbox_assigner = [
            MaxIoUAssigner(
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35),
            MaxIoUAssigner(
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35),
            MaxIoUAssigner(
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45)
        ]

        self.bbox_sampler = PseudoSampler()

        self.fp16_enabled = False

        #Initialize neural network layers of the head.
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                        self.num_anchors * 2, 1)

        self.loss_cls = FocalLoss(loss_weight=1.0)
        self.loss_bbox = SmoothL1Loss(beta=1.0 / 9.0, loss_weight=2.0)
        self.loss_dir = CrossEntropyLoss(loss_weight=0.2)


    def forward_single(self, x):
        """Forward function on a single-scale feature map.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox \
                regression and direction classification predictions.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_preds = None
        dir_cls_preds = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_preds


    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox \
                and direction predictions.
        """
        return multi_apply(
            self.forward_single,
            feats)


    def loss_single(self, cls_score, bbox_pred, dir_cls_preds, labels,
                    label_weights, bbox_targets, bbox_weights, dir_targets,
                    dir_weights, num_total_samples):
        """Calculate loss of Single-level results.

        Args:
            cls_score (torch.Tensor): Class score in single-level.
            bbox_pred (torch.Tensor): Bbox prediction in single-level.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single-level.
            labels (torch.Tensor): Labels of class.
            label_weights (torch.Tensor): Weights of class loss.
            bbox_targets (torch.Tensor): Targets of bbox predictions.
            bbox_weights (torch.Tensor): Weights of bbox loss.
            dir_targets (torch.Tensor): Targets of direction predictions.
            dir_weights (torch.Tensor): Weights of direction loss.
            num_total_samples (int): The number of valid samples.

        Returns:
            tuple[torch.Tensor]: Losses of class, bbox \
                and direction, respectively.
        """

        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1,1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        assert labels.max().item() <= self.num_classes
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
            
        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        # dir loss
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).reshape(-1, 2)
        dir_targets = dir_targets.reshape(-1)
        dir_weights = dir_weights.reshape(-1)
        pos_dir_cls_preds = dir_cls_preds[pos_inds]
        pos_dir_targets = dir_targets[pos_inds]
        pos_dir_weights = dir_weights[pos_inds]

        if num_pos > 0:
            pos_bbox_pred, pos_bbox_targets = self.add_sin_difference(
                pos_bbox_pred, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                pos_bbox_weights,
                avg_factor=num_total_samples)

            # direction classification loss
            loss_dir = self.loss_dir(
                pos_dir_cls_preds,
                pos_dir_targets,
                pos_dir_weights,
                avg_factor=num_total_samples)
        else:
            loss_bbox = pos_bbox_pred.sum()
            loss_dir = pos_dir_cls_preds.sum()

        return loss_cls, loss_bbox, loss_dir


    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th \
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2


    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             gt_bboxes,
             gt_labels):
        """Calculate losses.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Gt bboxes
                of each sample.
            gt_labels (list[torch.Tensor]): Gt labels of each sample.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and \
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification \
                    losses.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=cls_scores[0].device)
        anchor_list = [multi_level_anchors for _ in range(len(gt_labels))]
        
        label_channels = self.cls_out_channels
        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            gt_bboxes,
            gt_labels_list=gt_labels,
            num_classes=self.num_classes,
            label_channels=label_channels,
            sampling=False)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets

        losses_cls, losses_bbox, losses_dir = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            dir_targets_list,
            dir_weights_list,
            num_total_samples=num_total_pos)

        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir)


    def anchor_target_3d(self,
                         anchor_list,
                         gt_bboxes_list,
                         gt_bboxes_ignore_list=None,
                         gt_labels_list=None,
                         label_channels=1,
                         num_classes=1,
                         sampling=True):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            gt_bboxes_list (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                bboxes of each image.
            gt_bboxes_ignore_list (None | list): Ignore list of gt bboxes.
            gt_labels_list (list[torch.Tensor]): Gt labels of batches.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple (list, list, list, list, list, list, int, int):
                Anchor targets, including labels, label weights,
                bbox targets, bbox weights, direction targets,
                direction weights, number of postive anchors and
                number of negative anchors.
        """
        num_imgs = len(gt_labels_list)
        assert len(anchor_list) == num_imgs

        if isinstance(anchor_list[0][0], list):
            # sizes of anchors are different
            # anchor number of a single level
            num_level_anchors = [
                sum([anchor.size(0) for anchor in anchors])
                for anchors in anchor_list[0]
            ]
            for i in range(num_imgs):
                anchor_list[i] = anchor_list[i][0]
        else:
            # anchor number of multi levels
            num_level_anchors = [
                anchors.view(-1, self.box_code_size).size(0)
                for anchors in anchor_list[0]
            ]
            # concat all level anchors and flags to a single tensor
            for i in range(num_imgs):
                anchor_list[i] = torch.cat(anchor_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         all_dir_targets, all_dir_weights, pos_inds_list,
         neg_inds_list) = multi_apply(
             self.anchor_target_3d_single,
             anchor_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             label_channels=label_channels,
             num_classes=num_classes,
             sampling=sampling)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        dir_targets_list = images_to_levels(all_dir_targets, num_level_anchors)
        dir_weights_list = images_to_levels(all_dir_weights, num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, dir_targets_list, dir_weights_list,
                num_total_pos, num_total_neg)


    def anchor_target_single_assigner(self,
                                      bbox_assigner,
                                      anchors,
                                      gt_bboxes,
                                      gt_bboxes_ignore,
                                      gt_labels,
                                      num_classes=1,
                                      sampling=True):
        """Assign anchors and encode positive anchors.

        Args:
            bbox_assigner (BaseAssigner): assign positive and negative boxes.
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (torch.Tensor): Ignored gt bboxes.
            gt_labels (torch.Tensor): Gt class labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """
        anchors = anchors.reshape(-1, anchors.size(-1))
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        dir_targets = anchors.new_zeros((anchors.shape[0]), dtype=torch.long)
        dir_weights = anchors.new_zeros((anchors.shape[0]), dtype=torch.float)
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        if len(gt_bboxes) > 0:
            if not isinstance(gt_bboxes, torch.Tensor):
                gt_bboxes = gt_bboxes.tensor.to(anchors.device)
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
            sampling_result = self.bbox_sampler.sample(assign_result, anchors,
                                                       gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
        else:
            pos_inds = torch.nonzero(
                anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) > 0
            ).squeeze(-1).unique()
            neg_inds = torch.nonzero(
                anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) ==
                0).squeeze(-1).unique()

        if gt_labels is not None:
            labels += num_classes
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            pos_dir_targets = get_direction_target(
                sampling_result.pos_bboxes,
                pos_bbox_targets,
                0,
                one_hot=False)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dir_targets[pos_inds] = pos_dir_targets
            dir_weights[pos_inds] = 1.0

            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return (labels, label_weights, bbox_targets, bbox_weights, dir_targets,
                dir_weights, pos_inds, neg_inds)

    def anchor_target_3d_single(self,
                                anchors,
                                gt_bboxes,
                                gt_bboxes_ignore,
                                gt_labels,
                                label_channels=1,
                                num_classes=1,
                                sampling=True):
        """Compute targets of anchors in single batch.

        Args:
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (torch.Tensor): Ignored gt bboxes.
            gt_labels (torch.Tensor): Gt class labels.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """

        feat_size = anchors.size(0) * anchors.size(1) * anchors.size(2)
        rot_angles = anchors.size(-2)
        assert len(self.bbox_assigner) == anchors.size(-3)
        (total_labels, total_label_weights, total_bbox_targets,
            total_bbox_weights, total_dir_targets, total_dir_weights,
            total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
        current_anchor_num = 0
        for i, assigner in enumerate(self.bbox_assigner):
            current_anchors = anchors[..., i, :, :].reshape(
                -1, self.box_code_size)
            current_anchor_num += current_anchors.size(0)
            
            anchor_targets = self.anchor_target_single_assigner(
                assigner, current_anchors, gt_bboxes, gt_bboxes_ignore,
                gt_labels, num_classes, sampling)

            (labels, label_weights, bbox_targets, bbox_weights,
                dir_targets, dir_weights, pos_inds, neg_inds) = anchor_targets
            total_labels.append(labels.reshape(feat_size, 1, rot_angles))
            total_label_weights.append(
                label_weights.reshape(feat_size, 1, rot_angles))
            total_bbox_targets.append(
                bbox_targets.reshape(feat_size, 1, rot_angles,
                                        anchors.size(-1)))
            total_bbox_weights.append(
                bbox_weights.reshape(feat_size, 1, rot_angles,
                                        anchors.size(-1)))
            total_dir_targets.append(
                dir_targets.reshape(feat_size, 1, rot_angles))
            total_dir_weights.append(
                dir_weights.reshape(feat_size, 1, rot_angles))
            total_pos_inds.append(pos_inds)
            total_neg_inds.append(neg_inds)

        total_labels = torch.cat(total_labels, dim=-2).reshape(-1)
        total_label_weights = torch.cat(
            total_label_weights, dim=-2).reshape(-1)
        total_bbox_targets = torch.cat(
            total_bbox_targets, dim=-3).reshape(-1, anchors.size(-1))
        total_bbox_weights = torch.cat(
            total_bbox_weights, dim=-3).reshape(-1, anchors.size(-1))
        total_dir_targets = torch.cat(
            total_dir_targets, dim=-2).reshape(-1)
        total_dir_weights = torch.cat(
            total_dir_weights, dim=-2).reshape(-1)
        total_pos_inds = torch.cat(total_pos_inds, dim=0).reshape(-1)
        total_neg_inds = torch.cat(total_neg_inds, dim=0).reshape(-1)
        return (total_labels, total_label_weights, total_bbox_targets,
                total_bbox_weights, total_dir_targets, total_dir_weights,
                total_pos_inds, total_neg_inds)
        

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device

        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        mlvl_anchors = [
            anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors
        ]

        result_list = []
        for idx in range(len(cls_scores[0])):
            cls_score_list = [
                cls_scores[i][idx].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][idx].detach() for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls_preds[i][idx].detach() for i in range(num_levels)
            ]

            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               dir_cls_pred_list, mlvl_anchors)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          mlvl_anchors):
        """Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
                
        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)

            scores = cls_score.sigmoid()

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)

            nms_pre = 100
            if scores.shape[0] > nms_pre:  
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(LiDARInstance3DBoxes(
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        # Add a dummy background class to the front when using sigmoid
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = 0.1
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_scores, score_thr, 50, 
                                       mlvl_dir_scores)
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6], 1, np.pi)
            bboxes[..., 6] = (dir_rot + np.pi * dir_scores.to(bboxes.dtype))
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=self.box_code_size)
        return bboxes, scores, labels


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
        last_layer (bool): If last_layer, there is no concatenation of
            features.
        mode (str): Pooling model to gather features inside voxels.
            Default to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 last_layer=False,
                 mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        assert mode in ['max', 'avg']
        self.mode = mode

    #@auto_fp16(apply_to=('inputs'), out_fp32=True)
    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape (N, M, C).
                N is the number of voxels, M is the number of points in
                voxels, C is the number of channels of point features.
            num_voxels (torch.Tensor, optional): Number of points in each
                voxel. Defaults to None.
            aligned_distance (torch.Tensor, optional): The distance of
                each points to the voxel center. Defaults to None.

        Returns:
            torch.Tensor: Features of Pillars.
        """
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)

        if self.mode == 'max':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = x.sum(
                dim=1, keepdim=True) / num_voxels.type_as(inputs).view(
                    -1, 1, 1)

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNet(nn.Module):
    """Pillar Feature Net.

    The network prepares the pillar features and performs forward pass
    through PFNLayers.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        with_distance (bool, optional): Whether to include Euclidean distance
            to points. Defaults to False.
        with_cluster_center (bool, optional): [description]. Defaults to True.
        with_voxel_center (bool, optional): [description]. Defaults to True.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
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
                PFNLayer(
                    in_filters,
                    out_filters,
                    last_layer=last_layer, 
                    mode='max'))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    #@force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_points.type_as(features).view(
                -1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        
        f_center = features[:, :, :2]
        f_center[:, :, 0] = f_center[:, :, 0] - (
            coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
            self.x_offset)
        f_center[:, :, 1] = f_center[:, :, 1] - (
            coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
            self.y_offset)
            
        features_ls.append(f_center)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze()

class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, 
                 in_channels=64, 
                 output_shape=[496, 432]):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

    #@auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size=None):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        if batch_size is not None:
            return self.forward_batch(voxel_features, coors, batch_size)
        else:
            return self.forward_single(voxel_features, coors)

    def forward_single(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel.
                The first column indicates the sample ID.
        """
        # Create the canvas for this sample
        canvas = torch.zeros(
            self.in_channels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device)

        indices = coors[:, 1] * self.nx + coors[:, 2]
        indices = indices.long()
        voxels = voxel_features.t()
        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels
        # Undo the column stacking to final 4-dim tensor
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
        return [canvas]

    def forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny,
                                         self.nx)

        return batch_canvas