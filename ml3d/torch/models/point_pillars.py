#***************************************************************************************/
#
#    Based on MMDetection3D Library (Apache 2.0 license):
#    https://github.com/open-mmlab/mmdetection3d
#
#    Copyright 2018-2019 Open-MMLab.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#***************************************************************************************/

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from functools import partial
import numpy as np

from open3d.ml.torch.ops import voxelize, ragged_to_dense

from ...vis.boundingbox import BoundingBox3D

from .base_model_objdet import BaseModel

from ...utils import MODEL
from ..utils.objdet_helper import Anchor3DRangeGenerator, BBoxCoder, multiclass_nms, limit_period, get_paddings_indicator
from ..modules.losses.focal_loss import FocalLoss
from ..modules.losses.smooth_L1 import SmoothL1Loss
from ..modules.losses.cross_entropy import CrossEntropyLoss


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
        raise NotImplementedError

    def preprocess(self, data, attr):
        return data

    def transform(self, data, attr):
        #data = data['data']
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

    def inference_end(self, results):
        bboxes_b, scores_b, labels_b = self.bbox_head.get_bboxes(*results)

        inference_result = []

        for _bboxes, _scores, _labels in zip(bboxes_b, scores_b, labels_b):
            bboxes = _bboxes.cpu().numpy()
            scores = _scores.cpu().numpy()
            labels = _labels.cpu().numpy()
            inference_result.append([])

            for bbox, score, label in zip(bboxes, scores, labels):
                yaw = bbox[-1]
                cos = np.cos(yaw)
                sin = np.sin(yaw)

                front = np.array((sin, cos, 0))
                left = np.array((-cos, sin, 0))
                up = np.array((0, 0, 1))

                dim = bbox[[3, 5, 4]]
                pos = bbox[:3] + [0, 0, dim[1] / 2]

                inference_result[-1].append(
                    BoundingBox3D(pos, front, up, left, dim, label, score))

        return inference_result


MODEL._register_module(PointPillars, 'torch')


class PointPillarsVoxelization(torch.nn.Module):

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
        self.voxel_size = torch.Tensor(voxel_size)
        self.point_cloud_range = point_cloud_range
        self.points_range_min = torch.Tensor(point_cloud_range[:3])
        self.points_range_max = torch.Tensor(point_cloud_range[3:])

        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)

    def forward(self, points_feats):
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
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        points = points_feats[:, :3]

        ans = voxelize(points, self.voxel_size, self.points_range_min,
                                self.points_range_max, self.max_num_points,
                                max_voxels)

        # prepend row with zeros which maps to index 0 which maps to void points.
        feats = torch.cat(
            [torch.zeros_like(points_feats[0:1, :]), points_feats])

        # create dense matrix of indices. index 0 maps to the zero vector.
        voxels_point_indices_dense = ragged_to_dense(
            ans.voxel_point_indices, ans.voxel_point_row_splits,
            self.max_num_points, torch.tensor(-1)) + 1

        out_voxels = feats[voxels_point_indices_dense]
        out_coords = ans.voxel_coords[:, [2, 1, 0]].contiguous()
        out_num_points = ans.voxel_point_row_splits[
            1:] - ans.voxel_point_row_splits[:-1]

        return out_voxels, out_coords, out_num_points


class PFNLayer(nn.Module):
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
            x_max = x.sum(dim=1,
                          keepdim=True) / num_voxels.type_as(inputs).view(
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
            dim=1, keepdim=True) / num_points.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype

        f_center = features[:, :, :2].clone().detach()
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

    def __init__(self, in_channels=64, output_shape=[496, 432]):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

    #@auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size=None):
        """Forward function to scatter features."""
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
        canvas = torch.zeros(self.in_channels,
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
            canvas = torch.zeros(self.in_channels,
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


class SECOND(nn.Module):
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
            block = [
                nn.Conv2d(in_filters[i],
                          out_channels[i],
                          3,
                          bias=False,
                          stride=layer_strides[i],
                          padding=1),
                nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    nn.Conv2d(out_channels[i],
                              out_channels[i],
                              3,
                              bias=False,
                              padding=1))
                block.append(
                    nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
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
                upsample_layer = nn.Conv2d(in_channels=in_channels[i],
                                           out_channels=out_channel,
                                           kernel_size=stride,
                                           stride=stride,
                                           bias=False)

            deblock = nn.Sequential(
                upsample_layer,
                nn.BatchNorm2d(out_channel, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

    #@auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            torch.Tensor: Feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return out


class Anchor3DHead(nn.Module):

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
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        self.conv_dir_cls = nn.Conv2d(self.feat_channels, self.num_anchors * 2,
                                      1)

    def forward(self, x):
        """Forward function on a feature map.

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
        assert cls_scores.size()[-2:] == bbox_preds.size()[-2:]
        assert cls_scores.size()[-2:] == dir_preds.size()[-2:]

        anchors = self.anchor_generator.grid_anchors(cls_scores.shape[-2:],
                                                     device=cls_scores.device)
        anchors = anchors.reshape(-1, self.box_code_size)

        dir_preds = dir_preds.permute(1, 2, 0).reshape(-1, 2)
        dir_scores = torch.max(dir_preds, dim=-1)[1]

        cls_scores = cls_scores.permute(1, 2, 0).reshape(-1, self.num_classes)
        scores = cls_scores.sigmoid()

        bbox_preds = bbox_preds.permute(1, 2, 0).reshape(-1, self.box_code_size)

        if scores.shape[0] > self.nms_pre:
            max_scores, _ = scores.max(dim=1)
            _, topk_inds = max_scores.topk(self.nms_pre)
            anchors = anchors[topk_inds, :]
            bbox_preds = bbox_preds[topk_inds, :]
            scores = scores[topk_inds, :]
            dir_scores = dir_scores[topk_inds]

        bboxes = self.bbox_coder.decode(anchors, bbox_preds)

        idxs = multiclass_nms(bboxes, scores, self.score_thr)

        labels = [
            torch.full((len(idxs[i]),), i, dtype=torch.long)
            for i in range(self.num_classes)
        ]
        labels = torch.cat(labels)

        scores = [scores[idxs[i], i] for i in range(self.num_classes)]
        scores = torch.cat(scores)

        idxs = torch.cat(idxs)
        bboxes = bboxes[idxs]
        dir_scores = dir_scores[idxs]

        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6], 1, np.pi)
            bboxes[..., 6] = (dir_rot + np.pi * dir_scores.to(bboxes.dtype))
            
        return bboxes, scores, labels
