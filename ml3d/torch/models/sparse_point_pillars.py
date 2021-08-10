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
import pickle
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from functools import partial
import numpy as np
import os

from open3d.ml.torch.ops import voxelize, ragged_to_dense
import MinkowskiEngine as ME

from .base_model_objdet import BaseModel
from .point_pillars import PointPillars, PointPillarsVoxelization, PointPillarsScatter, PillarFeatureNet, SECOND, SECONDFPN, Anchor3DHead

from ...utils import MODEL
from ..utils.objdet_helper import Anchor3DRangeGenerator, BBoxCoder, multiclass_nms, limit_period, get_paddings_indicator, bbox_overlaps, box3d_to_bev2d
from ..modules.losses.focal_loss import FocalLoss
from ..modules.losses.smooth_L1 import SmoothL1Loss
from ..modules.losses.cross_entropy import CrossEntropyLoss
from ...datasets.utils import ObjdetAugmentation, BEVBox3D
from ...datasets.utils.operations import filter_by_min_points


class SparsePointPillars(PointPillars):
    """Object detection model. Based on the PointPillars architecture
    https://github.com/nutonomy/second.pytorch.

    Args:
        name (string): Name of model.
            Default to "SparsePointPillars".
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
                 name="SparsePointPillars",
                 device="cuda",
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
                         device=device,
                         point_cloud_range=point_cloud_range,
                         classes=classes, 
                         voxelize=voxelize,
                         voxel_encoder=voxel_encoder,
                         scatter=scatter,
                         backbone=backbone,
                         neck=neck,
                         head=head,
                         loss=loss,
                         **kwargs)
        self.middle_encoder = SparsePointPillarsScatter(**scatter)
        self.backbone = SparseSECOND(**backbone)
        self.neck = SparseSECONDFPN(**neck)
        self.bbox_head = Anchor3DHead(num_classes=len(self.classes), **head)
        self.to(self.device)


MODEL._register_module(SparsePointPillars, 'torch')

class SparsePointPillarsScatter(nn.Module):
    """Sparse Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image in 
    sparse tensor format.

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
    def forward(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.
        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        sparse_coords = coors[:, [0,2,3]]
        # print(sparse_coords.shape)
        # print(voxel_features.shape)
        out_shape = (batch_size, self.ny, self.nx, voxel_features.shape[1])
        # print(out_shape)
        sp_batch = torch.sparse_coo_tensor(sparse_coords.t(), voxel_features, out_shape)
        return sp_batch


class SparseSECOND(nn.Module):
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
        super(SparseSECOND, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                ME.MinkowskiConvolution(in_filters[i],
                          out_channels[i],
                          3,
                          bias=False,
                          stride=layer_strides[i], 
                          dimension=2),
                ME.MinkowskiBatchNorm(out_channels[i], eps=1e-3, momentum=0.01),
                ME.MinkowskiReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    ME.MinkowskiConvolution(out_channels[i],
                              out_channels[i],
                              3,
                              bias=False,
                              dimension=2))
                block.append(
                    ME.MinkowskiBatchNorm(out_channels[i], eps=1e-3, momentum=0.01))
                block.append(ME.MinkowskiReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.layer_strides = layer_strides

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        input_shape = x.shape
        vals = x._values()
        idxs = x._indices().permute(1, 0).contiguous().int()
        x = ME.SparseTensor(vals, idxs)
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return input_shape, self.layer_strides, tuple(outs)


class ToDenseMink(nn.Module):
    def __init__(self, input_shape, first_shrink_stride, first_upsample_stride, out_size):
        super(ToDenseMink, self).__init__()
        batch_size, x_size, y_size, _ = input_shape
        scale = first_shrink_stride // first_upsample_stride
        self.output_shape = torch.Size([batch_size, out_size, x_size // scale, y_size // scale])
        self.min_coord = torch.IntTensor([0, 0])
    
    def forward(self, x):
        return x.dense(shape=self.output_shape, min_coordinate=self.min_coord)[0]

class SparseSECONDFPN(nn.Module):
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
        super(SparseSECONDFPN, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = ME.MinkowskiConvolutionTranspose(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                    bias=False, 
                    dimension=2)
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = ME.MinkowskiConvolution(in_channels=in_channels[i],
                                           out_channels=out_channel,
                                           kernel_size=stride,
                                           stride=stride,
                                           bias=False, 
                                           dimension=2)

            deblock = nn.Sequential(
                upsample_layer,
                ME.MinkowskiBatchNorm(out_channel, eps=1e-3, momentum=0.01),
                ME.MinkowskiReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
        self.init_weights()
        self.upsample_strides = upsample_strides

    def init_weights(self):
        """Initialize weights of FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            torch.Tensor: Feature maps.
        """
        input_shape, layer_strides, x = x
        assert len(x) == len(self.in_channels)
        ups = [ToDenseMink(input_shape, 
                           layer_strides[0], 
                           self.upsample_strides[0], 
                           self.out_channels[i])(deblock(x[i]))
            for i, deblock in enumerate(self.deblocks)]
        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return out

# class SparseAnchor3DHead(Anchor3DHead):

#     def __init__(self,
#                  num_classes=1,
#                  in_channels=384,
#                  feat_channels=384,
#                  nms_pre=100,
#                  score_thr=0.1,
#                  dir_offset=0,
#                  ranges=[[0, -40.0, -3, 70.0, 40.0, 1]],
#                  sizes=[[0.6, 1.0, 1.5]],
#                  rotations=[0, 1.57],
#                  iou_thr=[[0.35, 0.5]]):

#         super().__init__(num_classes=num_classes,
#                          in_channels=in_channels,
#                          feat_channels=feat_channels,
#                          nms_pre=nms_pre,
#                          score_thr=score_thr,
#                          dir_offset=dir_offset,
#                          ranges=ranges,
#                          sizes=sizes,
#                          rotations=rotations,
#                          iou_thr=iou_thr)
        
#         self.conv_cls = ME.MinkowskiConvolution(self.feat_channels, self.cls_out_channels, 1, dimension=2)
#         self.conv_reg = ME.MinkowskiConvolution(self.feat_channels,
#                                   self.num_anchors * self.box_code_size, 1, dimension=2)
#         self.conv_dir_cls = ME.MinkowskiConvolution(self.feat_channels, self.num_anchors * 2,
#                                       1, dimension=2)

#         self.init_weights()

