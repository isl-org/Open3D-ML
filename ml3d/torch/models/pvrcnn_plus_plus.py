from turtle import down, shape
from matplotlib.transforms import BboxBase
from numpy import pad
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_
from functools import partial
import numpy as np
import math

from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.config import *

# from OpenPCDet.pcdet.utils.box_utils import enlarge_box3d

from .base_model_objdet import BaseModel
import open3d.ml.torch as ml3d
from open3d.ml.torch.ops import voxelize, ragged_to_dense, reduce_subarrays_sum, nms
from open3d.ml.torch.layers import SparseConv
from ..utils.objdet_helper import multiclass_nms, box3d_to_bev, xywhr_to_xyxyr, bbox_overlaps
from ...utils import MODEL



class PVRCNNPlusPlus(BaseModel):
    def __init__(self, 
                 name = "PVRCNNPlusPlus",
                 device="cuda",
                 point_cloud_range=[0, -40.0, -3, 70.0, 40.0, 1],
                 classes=['car'],
                 voxelize={},
                 voxel_encoder={},
                 backbone_2d = {},
                 rpn_module = {},
                 voxel_set_abstraction = {},
                 keypoint_weights = {},
                 **kwargs):
        super().__init__(name=name,
                         point_cloud_range=point_cloud_range,
                         device=device,
                         **kwargs)
        self.point_cloud_range = point_cloud_range
        self.classes = classes
        self.device = device
        self.name2lbl = {n: i for i, n in enumerate(classes)}
        self.to(device)
        self.voxel_layer = PVRCNNPlusPlusVoxelization(point_cloud_range=point_cloud_range, **voxelize)
        self.sparseconvbackbone = PVRCNNPlusPlusBackbone3D(point_cloud_range=point_cloud_range, **voxel_encoder)
        self.bev_module = PVRCNNPlusPlusBEVModule(point_cloud_range=point_cloud_range)
        self.backbone_2d = PVRCNNPlusPlusBackbone2D(640, **backbone_2d)
        self.rpn_module = RPNModule(input_channels=512, device = device, point_cloud_range=point_cloud_range, **rpn_module)
        self.voxel_set_abstraction = PVRCNNPlusPlusVoxelSetAbstraction(point_cloud_range=point_cloud_range, **voxel_set_abstraction)
        self.keypoint_weight_computer = PVRCNNKeypointWeightComputer(num_class = len(self.classes), input_channels = 90, **keypoint_weights)#cls_fc, gt_extra_width)
        self.box_refinement = PVRCNNPlusPlusBoxRefinement(num_class = len(self.classes), input_channels = 90, code_size = 7)
        self.device = device
        self.to(device)

    @torch.no_grad()
    def voxelize(self, points):
        """Apply voxelization to points."""
        voxels, coors, num_points, voxel_features = [], [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
            points_mean = res_voxels[:, :, :].sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(res_num_points.view(-1, 1), min=1.0).type_as(res_voxels)
            points_mean = points_mean / normalizer
            voxel_features.append(points_mean.contiguous())
        # voxels = torch.cat(voxels, dim=0)
        # voxel_features = torch.cat(voxels, dim=0)
        # num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        # coors_batch = torch.cat(coors_batch, dim=0)

        return voxels, voxel_features, num_points, coors, coors_batch
    
    def convert_open3d_voxel_to_dense(self, x_open3d, x_pos_open3d, voxel_size = 1.0):
        x_sparse = []
        self.point_cloud_range = np.array(self.point_cloud_range)
        for x, x_pos in zip(x_open3d, x_pos_open3d):
            dense_shape = ((self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / (np.array(voxel_size)*8) + 1).astype(int)[::-1].tolist()
            dense_shape.append(x.shape[1])
            x_sparse_temp = torch.sparse_coo_tensor(torch.tensor(x_pos).t(), x, dense_shape)
            x_sparse.append(x_sparse_temp)
        return x_sparse

    def backbone_3d(self, points):
        voxels, voxel_features, num_points, coors, coors_batch = self.voxelize(points)
        x_intermediate_layers, x, x_pos = self.sparseconvbackbone(voxel_features, coors)
        return x, x_intermediate_layers, x_pos

    def forward(self, inputs):
        gt_boxes = inputs.bboxes
        gt_labels = inputs.labels
        print(gt_boxes[0].shape, gt_labels[0].shape, torch.cat([gt_boxes[0], gt_labels[0].view(-1,1)], dim=1).shape)
        inputs = inputs.point
        x, x_intermediate_layers, x_pos = self.backbone_3d(inputs)
        x_sparse = self.convert_open3d_voxel_to_dense(x, x_pos, self.voxel_layer.voxel_size)
        x_dense_bev = self.bev_module(x_sparse)
        print("Shape of downsampled output", x_dense_bev[0].shape)
        x_bev_2d = self.backbone_2d(x_dense_bev)
        print("Shape of BEV-2D", x_bev_2d.shape)
        rois, roi_scores, roi_labels = self.rpn_module(x_bev_2d)
        print("Shape of ROIs", rois.shape)
        point_features, point_coords, point_features_before_fusion = self.voxel_set_abstraction(inputs, rois, x_bev_2d, x_intermediate_layers) 
        print("Shape of point features", point_features.shape)
        keypoint_weights = self.keypoint_weight_computer(len(gt_boxes), point_features, point_coords, point_features_before_fusion, gt_boxes, gt_labels)
        print("Shape of keypoint weights", keypoint_weights.shape)
        outs = self.box_refinement(point_coords, point_features, rois, roi_scores, roi_labels, gt_boxes, gt_labels, keypoint_weights)
        print(outs["rcnn_cls"].shape, outs["rcnn_reg"].shape)
        return x, x_intermediate_layers, x_pos
    
    def get_optimizer(self, cfg):
        pass

    def get_loss(self, results, inputs):
        pass

    def preprocess(self, data, attr):
        # If num_workers > 0, use new RNG with unique seed for each thread.
        # Else, use default RNG.
        if torch.utils.data.get_worker_info():
            seedseq = np.random.SeedSequence(
                torch.utils.data.get_worker_info().seed +
                torch.utils.data.get_worker_info().id)
            rng = np.random.default_rng(seedseq.spawn(1)[0])
        else:
            rng = self.rng

        points = np.array(data['point'][:, 0:4], dtype=np.float32)

        min_val = np.array(self.point_cloud_range[:3])
        max_val = np.array(self.point_cloud_range[3:])

        points = points[np.where(
            np.all(np.logical_and(points[:, :3] >= min_val,
                                  points[:, :3] < max_val),
                   axis=-1))]

        data['point'] = points

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

    def transform(self, data, attr):
        t_data = {'point': data['point'], 'calib': data['calib']}

        if attr['split'] not in ['test', 'testing']:
            t_data['bbox_objs'] = data['bbox_objs']
            t_data['labels'] = np.array([
                self.name2lbl.get(bb.label_class, len(self.classes))
                for bb in data['bbox_objs']
            ],
                                        dtype=np.int64)
            t_data['bboxes'] = np.array(
                [bb.to_xyzwhlr() for bb in data['bbox_objs']], dtype=np.float32)

        return t_data

    def inference_end(self, results, inputs):
        return results

"""
Voxelization Layer
Mean in OpenPCDet implementation
"""
class PVRCNNPlusPlusVoxelization(nn.Module):
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points=32,
                 max_voxels=[16000, 40000]):
        """Voxelization layer for the PVRCNN++ model.

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
        if isinstance(max_voxels, tuple) or isinstance(max_voxels, list):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        pass

    def forward(self, points_feats):
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
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        points = points_feats[:, :3]

        ans = voxelize(points,
                       torch.LongTensor([0, points.shape[0]]).to(points.device),
                       self.voxel_size, self.points_range_min,
                       self.points_range_max, self.max_num_points, max_voxels)
        
        # create dense matrix of indices. index 0 maps to the zero vector.
        voxels_point_indices_dense = ragged_to_dense(
            ans.voxel_point_indices, ans.voxel_point_row_splits,
            self.max_num_points, torch.tensor(-1)) + 1

        out_voxels = points_feats[voxels_point_indices_dense]
        out_coords = ans.voxel_coords[:, [2, 1, 0]].contiguous()
        out_num_points = ans.voxel_point_row_splits[
            1:] - ans.voxel_point_row_splits[:-1]
        
        return out_voxels, out_coords, out_num_points

class SequentialSparseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, point_cloud_range, voxel_size, padding, norm_fn, downsample = False):
        super().__init__()
        self.conv1 = SparseConvBlock(in_channels, out_channels, kernel_size, point_cloud_range, voxel_size, padding=1, norm_fn=norm_fn, downsample = downsample)
        self.conv2 = SubmanifoldSparseConvBlock(out_channels, out_channels, kernel_size, norm_fn=norm_fn)
        self.conv3 = SubmanifoldSparseConvBlock(out_channels, out_channels, kernel_size, norm_fn=norm_fn)
    
    def forward(self, features_list, in_positions_list):
        x_conv1, x_conv1_pos = self.conv1(features_list, in_positions_list)
        x_conv2, x_conv2_pos = self.conv2(x_conv1, x_conv1_pos)
        x_conv3, x_conv3_pos = self.conv3(x_conv2, x_conv2_pos)

        return x_conv3, x_conv3_pos

"""
Layer to obtain outputs from different layers of 3D Sparse Convolution Network
Will also need to store the intermediate layer outputs
"""
class PVRCNNPlusPlusBackbone3D(nn.Module):
    def __init__(self, in_channels, point_cloud_range, voxel_size):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        norm_fn = norm_fn = partial(BatchNormBlock, eps=1e-3, momentum=0.01)

        self.conv_input = SubmanifoldSparseConvBlock(in_channels, 16, [3,3,3], norm_fn = norm_fn)

        self.conv1 = SparseConvBlock(16, 16, [3,3,3], point_cloud_range, voxel_size, padding=1, norm_fn=norm_fn)
        
        # self.conv2 = nn.Sequential(
        #     SparseConvBlock(16, 32, [3,3,3], point_cloud_range, voxel_size, padding=1, norm_fn=norm_fn), #stride=2
        #     SubmanifoldSparseConvBlock(32, 32, [3,3,3], norm_fn=norm_fn),
        #     SubmanifoldSparseConvBlock(32, 32, [3,3,3], norm_fn=norm_fn),
        # )
        self.conv2 = SequentialSparseConv(16, 32, [3,3,3], point_cloud_range, voxel_size, padding=1, norm_fn=norm_fn, downsample = True)

        # self.conv3 = nn.Sequential(
        #     SparseConvBlock(32, 64, [3,3,3], point_cloud_range, voxel_size, padding=1, norm_fn=norm_fn), #stride=2
        #     SubmanifoldSparseConvBlock(64, 64, [3,3,3], norm_fn=norm_fn),
        #     SubmanifoldSparseConvBlock(64, 64, [3,3,3], norm_fn=norm_fn),
        # )
        self.conv3 = SequentialSparseConv(32, 64, [3,3,3], point_cloud_range, voxel_size, padding=1, norm_fn=norm_fn, downsample = True)

        # self.conv4 = nn.Sequential(
        #     SparseConvBlock(64, 64, [3,3,3], point_cloud_range, voxel_size, padding=1, norm_fn=norm_fn), #stride=2, padding = (0, 1, 1)
        #     SubmanifoldSparseConvBlock(64, 64, [3,3,3], norm_fn=norm_fn),
        #     SubmanifoldSparseConvBlock(64, 64, [3,3,3], norm_fn=norm_fn),
        # )
        self.conv4 = SequentialSparseConv(64, 64, [3,3,3], point_cloud_range, voxel_size, padding=1, norm_fn=norm_fn, downsample = True)

        self.conv_out = SparseConvBlock(64, 128, [3,3,3], point_cloud_range, voxel_size, padding = 0, norm_fn = norm_fn) #stride = (2, 1, 1)


    def forward(self, voxel_features, coors):
        x, x_pos = self.conv_input(voxel_features, coors)
        x_conv1, x_conv1_pos = self.conv1(x, x_pos)
        x_conv2, x_conv2_pos = self.conv2(x_conv1, x_conv1_pos)
        x_conv3, x_conv3_pos = self.conv3(x_conv2, x_conv2_pos)
        x_conv4, x_conv4_pos = self.conv4(x_conv3, x_conv3_pos)

        out, out_pos = self.conv_out(x_conv4, x_conv4_pos)

        layers_dict = {}
        layers_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv1_pos': x_conv1_pos,
                'x_conv2_pos': x_conv2_pos,
                'x_conv3_pos': x_conv3_pos,
                'x_conv4_pos': x_conv4_pos,
            }
        })

        return layers_dict, out, out_pos

"""
Convert the Last layer of 3D Sparse Convolution network to Birds Eye-View via Height compression
"""
class PVRCNNPlusPlusBEVModule(nn.Module):
    def __init__(self, point_cloud_range, method = "HeightCompression"):
        super().__init__()
        self.method = method
        self.point_cloud_range = point_cloud_range
    
    def forward(self, sparse_tensor_list):
        x_dense_list = []
        for x_sparse in sparse_tensor_list:
            spatial_features = x_sparse.to_dense().permute(3, 0, 1, 2)
            C, D, H, W= spatial_features.shape
            spatial_features = spatial_features.contiguous().view(C * D, H, W)
            x_dense_list.append(spatial_features)
        if len(x_dense_list) > 1:
            x_dense = torch.cat(x_dense_list, dim=0)
        else:
            x_dense = torch.cat(x_dense_list, dim=0).unsqueeze(dim = 0)
        return x_dense

class PVRCNNPlusPlusBackbone2D(nn.Module):
    def __init__(self, input_channels, layer_nums, layer_strides, num_filters, num_upsample_filters, upsample_strides):
        super().__init__()
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        self.num_bev_features = c_in
    
    def forward(self, spatial_features):
        ups = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        return x

class RPNHead(nn.Module):
    def __init__(self, shared_conv_channels, output_channels, num_head_conv):
        super().__init__()
        conv_list = []
        for k in range(num_head_conv - 1):
            conv_list.append(nn.Sequential(
                nn.Conv2d(shared_conv_channels, shared_conv_channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(shared_conv_channels),
                nn.ReLU()
            ))
        conv_list.append(nn.Conv2d(shared_conv_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv_head = nn.Sequential(*conv_list)

        for m in self.conv_head.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_normal_(m.weight.data)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv_head(x)

"""
Proposes and classifies 3D Bounding Boxes
one head for the following : center - 2, center_z - 1, dim - 3, rot - 2 and class
Applies NMS here as well
Outputs rois, roi_scores, roi_labels, has_class_labels
"""
class RPNModule(nn.Module):
    def __init__(self, input_channels, device, shared_conv_channels, num_head_conv, class_names, point_cloud_range, voxel_size, feature_map_stride, nms_pre, score_thr, post_center_limit_range):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = feature_map_stride
        self.class_names = class_names
        self.num_head_conv = num_head_conv
        self.nms_pre = nms_pre
        self.score_thr = score_thr
        self.post_center_limit_range = torch.tensor(post_center_limit_range).to(device)

        self.class_id_mapping = torch.from_numpy(np.array(
            [self.class_names.index(x) for x in class_names]
        )).cuda()
        
        total_classes = len(self.class_names)

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, shared_conv_channels, 3, stride=1, padding=1,
                bias=True
            ),
            nn.BatchNorm2d(shared_conv_channels),
            nn.ReLU(),
        )
        self.heads_dict = nn.ModuleList()
        #Creating seperate heads for Center, Center_z, Dim, Rot, classification
        #Center - self.heads_list[0]
        conv_head = RPNHead(shared_conv_channels, 2, num_head_conv)
        self.heads_dict.append(conv_head)

        #Center_z - self.heads_list[1]
        conv_head = RPNHead(shared_conv_channels, 1, num_head_conv)
        self.heads_dict.append(conv_head)
        
        #Dim - self.heads_list[2]
        conv_head = RPNHead(shared_conv_channels, 3, num_head_conv)
        self.heads_dict.append(conv_head)
        
        #Rot - self.heads_list[3]
        conv_head = RPNHead(shared_conv_channels, 2, num_head_conv)
        self.heads_dict.append(conv_head)
        
        #classification - self.head_list[4]
        conv_head = RPNHead(shared_conv_channels, total_classes, num_head_conv)
        self.heads_dict.append(conv_head)
    
    def forward(self, bev_feats):
        x = self.shared_conv(bev_feats)
        preds = []
        for head in self.heads_dict:
            preds.append(head(x))
        
        batch_size = preds[4].size()[0]

        bboxes, scores, labels = self.generate_predicted_boxes(batch_size, preds)
        rois, roi_scores, roi_labels = self.reorder_rois_for_refining(batch_size, bboxes, scores, labels)

        return rois, roi_scores, roi_labels

    def center_to_edge_bbox(self, bboxes):
        bboxes[:,:3] = bboxes[:,:3] - (bboxes[:,3:6]/2)
        return bboxes 

    def generate_predicted_boxes(self, batch_size, preds):
        bboxes, scores, labels = [], [], []
        batch_center = preds[0]
        batch_center_z = preds[1]
        batch_dim = preds[2].exp()
        batch_rot_cos = preds[3][:, 0].unsqueeze(dim=1)
        batch_rot_sin = preds[3][:, 1].unsqueeze(dim=1)
        batch_hm = preds[4].sigmoid()
        final_pred_dicts = self.decode_bbox_from_heatmap(
            heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
            center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=None,
            point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
            feature_map_stride=self.feature_map_stride,
            K=self.nms_pre,
            score_thresh=self.score_thr,
            post_center_limit_range=self.post_center_limit_range
            )

        for k, final_dict in enumerate(final_pred_dicts):
            bbox_pred = self.center_to_edge_bbox(final_dict["pred_boxes"])
            score_pred = final_dict["pred_scores"].view(-1).sigmoid()
            label_pred = final_dict["pred_labels"]
            idxs = self.multiclass_nms(bbox_pred, score_pred, label_pred, self.score_thr)
            idxs = torch.cat(idxs)
            final_dict["pred_boxes"] = final_dict["pred_boxes"][idxs]
            final_dict["pred_labels"] = final_dict["pred_labels"][idxs]
            final_dict["pred_scores"] = final_dict["pred_scores"][idxs]
            bboxes.append(final_dict["pred_boxes"])
            labels.append(final_dict["pred_labels"])
            scores.append(final_dict["pred_scores"])
        
        return bboxes, scores, labels

    def gaussian_radius(self, height, width, min_overlap=0.5):
        """
        Args:
            height: (N)
            width: (N)
            min_overlap:
        Returns:
        """
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
        r3 = (b3 + sq3) / 2
        ret = torch.min(torch.min(r1, r2), r3)
        return ret

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h
    
    def draw_gaussian_to_heatmap(self, heatmap, center, radius, k=1, valid_mask=None):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = torch.from_numpy(
            gaussian[radius - top:radius + bottom, radius - left:radius + right]
        ).to(heatmap.device).float()

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            if valid_mask is not None:
                cur_valid_mask = valid_mask[y - top:y + bottom, x - left:x + right]
                masked_gaussian = masked_gaussian * cur_valid_mask.float()

            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap
    
    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep
    
    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _topk(self, scores, K=40):
        batch, num_class, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.flatten(2, 3), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_classes = (topk_ind // K).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    
        return topk_score, topk_inds, topk_classes, topk_ys, topk_xs
    
    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat
    
    def multiclass_nms(self, boxes, scores, class_ids, score_thr):
        #class based nms
        idxs = []
        for i in range(class_ids.max()+1):
            class_mask = class_ids==i
            orig_idx = torch.arange(class_mask.shape[0],
                                device=class_mask.device,
                                dtype=torch.long)[class_mask]
            _scores = scores[class_mask]
            _boxes = boxes[class_mask, :]
            _bev = xywhr_to_xyxyr(box3d_to_bev(_boxes))
            idx = nms(_bev, _scores, 0.01)
            idxs.append(orig_idx[idx])
        
        return idxs

    def decode_bbox_from_heatmap(self, heatmap, rot_cos, rot_sin, center, center_z, dim,
                             point_cloud_range=None, voxel_size=None, feature_map_stride=None, vel=None, K=100, 
                             score_thresh=None, post_center_limit_range=None):
        batch_size, num_class, _, _ = heatmap.size()

        scores, inds, class_ids, ys, xs = self._topk(heatmap, K=K)
        center = self._transpose_and_gather_feat(center, inds).view(batch_size, K, 2)
        rot_sin = self._transpose_and_gather_feat(rot_sin, inds).view(batch_size, K, 1)
        rot_cos = self._transpose_and_gather_feat(rot_cos, inds).view(batch_size, K, 1)
        center_z = self._transpose_and_gather_feat(center_z, inds).view(batch_size, K, 1)
        dim = self._transpose_and_gather_feat(dim, inds).view(batch_size, K, 3)

        angle = torch.atan2(rot_sin, rot_cos)
        xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]

        xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
        ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

        box_part_list = [xs, ys, center_z, dim, angle]
        if vel is not None:
            vel = self._transpose_and_gather_feat(vel, inds).view(batch_size, K, 2)
            box_part_list.append(vel)

        final_box_preds = torch.cat((box_part_list), dim=-1)
        final_scores = scores.view(batch_size, K)
        final_class_ids = class_ids.view(batch_size, K)

        # assert post_center_limit_range is not None
        # mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
        # mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)

        # if score_thresh is not None:
        #     mask = (final_scores > score_thresh)

        ret_pred_dicts = []
        for k in range(batch_size):
            # cur_mask = mask[k]
            cur_boxes = final_box_preds[k]#, cur_mask]
            cur_scores = final_scores[k]#, cur_mask]
            cur_labels = final_class_ids[k]#, cur_mask]

            ret_pred_dicts.append({
                'pred_boxes': cur_boxes,
                'pred_scores': cur_scores,
                'pred_labels': cur_labels
            })
        return ret_pred_dicts
    
    @staticmethod
    def reorder_rois_for_refining(batch_size, bboxes, scores, labels):
        num_max_rois = max([len(bbox) for bbox in bboxes])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = bboxes[0]

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(bboxes[bs_idx])

            rois[bs_idx, :num_boxes, :] = bboxes[bs_idx]
            roi_scores[bs_idx, :num_boxes] = scores[bs_idx]
            roi_labels[bs_idx, :num_boxes] = labels[bs_idx]
        return rois, roi_scores, roi_labels

"""
Used inside PVRSCNNPlusPlusVoxelSetabstraction
"""
class PVRCNNPlusPlusVectorPoolAggregationModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, roi, point_coords, point_feats, keypoints):
        pass

"""
Returns points_features_before_fusion and point_features and point_coords and uses the VectorPoolAgregation module
"""
class PVRCNNPlusPlusVoxelSetAbstraction(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, num_bev_features, num_raw_features, num_keypoints, sample_radius_with_roi, num_sectors, **kwargs):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.num_keypoints = num_keypoints
        self.sample_radius_with_roi = sample_radius_with_roi
        self.num_sectors = num_sectors
        c_in = 0

        config = cfg_from_yaml_file("/gsoc/OpenPCDet/tools/cfgs/waymo_models/pv_rcnn_plusplus.yaml", cfg)
        self.model_cfg = cfg.MODEL.PFE
        SA_cfg = self.model_cfg.SA_LAYER
        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []

        #Raw Points
        self.SA_rawpoints, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=num_raw_features - 3, config=SA_cfg['raw_points']
        )
        c_in += cur_num_c_out

        #BEV
        c_bev = num_bev_features
        c_in += c_bev

        #x_conv_3
        src_name = "x_conv3"
        input_channels = 64
        cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=SA_cfg[src_name]
        )
        self.SA_layers.append(cur_layer)
        self.SA_layer_names.append(src_name)
        c_in += cur_num_c_out

        # #x_conv_4
        # src_name = "x_conv_4"
        # input_channels = 64
        # cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
        #     input_channels=input_channels, config=SA_cfg[src_name]
        # )
        # self.SA_layers.append(cur_layer)
        # self.SA_layer_names.append(src_name)
        # c_in += cur_num_c_out

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )

        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

    def get_voxel_centers(self, voxel_coords, downsample_times, voxel_size, point_cloud_range):
        """
        Args:
            voxel_coords: (N, 3)
            downsample_times:
            voxel_size:
            point_cloud_range:

        Returns:

        """
        assert voxel_coords.shape[1] == 3
        voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
        voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
        pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
        voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
        return voxel_centers

    def sample_points_with_roi(self, rois, points, sample_radius_with_roi, num_max_points_of_part=200000):
        if points.shape[0] < num_max_points_of_part:
            distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            point_mask = min_dis < roi_max_dim + sample_radius_with_roi
        else:
            start_idx = 0
            point_mask_list = []
            while start_idx < points.shape[0]:
                distance = (points[start_idx:start_idx + num_max_points_of_part, None, :] - rois[None, :, 0:3]).norm(dim=-1)
                min_dis, min_dis_roi_idx = distance.min(dim=-1)
                roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
                cur_point_mask = min_dis < roi_max_dim + sample_radius_with_roi
                point_mask_list.append(cur_point_mask)
                start_idx += num_max_points_of_part
            point_mask = torch.cat(point_mask_list, dim=0)

        sampled_points = points[:1] if point_mask.sum() == 0 else points[point_mask, :]

        return sampled_points, point_mask

    def sector_fps(self, points, num_sampled_points, num_sectors):
        sector_size = np.pi * 2 / num_sectors
        point_angles = torch.atan2(points[:, 1], points[:, 0]) + np.pi
        sector_idx = (point_angles / sector_size).floor().clamp(min=0, max=num_sectors)
        xyz_points_list = []
        xyz_batch_cnt = []
        num_sampled_points_list = []
        for k in range(num_sectors):
            mask = (sector_idx == k)
            cur_num_points = mask.sum().item()
            if cur_num_points > 0:
                xyz_points_list.append(points[mask])
                xyz_batch_cnt.append(cur_num_points)
                ratio = cur_num_points / points.shape[0]
                num_sampled_points_list.append(
                    min(cur_num_points, math.ceil(ratio * num_sampled_points))
                )

        if len(xyz_batch_cnt) == 0:
            xyz_points_list.append(points)
            xyz_batch_cnt.append(len(points))
            num_sampled_points_list.append(num_sampled_points)
            print(f'Warning: empty sector points detected in SectorFPS: points.shape={points.shape}')

        xyz = torch.cat(xyz_points_list, dim=0)
        xyz_batch_cnt = torch.tensor(xyz_batch_cnt, device=points.device).int()
        sampled_points_batch_cnt = torch.tensor(num_sampled_points_list, device=points.device).int()

        sampled_pt_idxs = pointnet2_stack_utils.stack_farthest_point_sample(
            xyz.contiguous(), xyz_batch_cnt, sampled_points_batch_cnt
        ).long()

        sampled_points = xyz[sampled_pt_idxs]

        return sampled_points

    def sectorized_proposal_centric_sampling(self, roi_boxes, points):
        sampled_points, _ = self.sample_points_with_roi(
            rois=roi_boxes, points=points,
            sample_radius_with_roi=self.sample_radius_with_roi
        )
        sampled_points = self.sector_fps(
            points=sampled_points, num_sampled_points=self.num_keypoints,
            num_sectors=self.num_sectors
        )
        return sampled_points

    def get_keypoints(self, points, bboxes, batch_size):
        keypoints_list = []
        for bs_idx in range(batch_size):
            sampled_points = points[bs_idx][:, :3].view(1,-1,3)
            cur_keypoints = self.sectorized_proposal_centric_sampling(
                    roi_boxes=bboxes[bs_idx], points=sampled_points[0]
                )
            bs_idxs = cur_keypoints.new_ones(cur_keypoints.shape[0]) * bs_idx
            keypoints = torch.cat((bs_idxs[:, None], cur_keypoints), dim=1)
            keypoints_list.append(keypoints)
        
        keypoints = torch.cat(keypoints_list, dim=0)
        return keypoints
    
    def bilinear_interpolate_torch(self, im, x, y):
        x0 = torch.floor(x).long()
        x1 = x0 + 1

        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, im.shape[1] - 1)
        x1 = torch.clamp(x1, 0, im.shape[1] - 1)
        y0 = torch.clamp(y0, 0, im.shape[0] - 1)
        y1 = torch.clamp(y1, 0, im.shape[0] - 1)

        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]

        wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
        wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
        wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
        wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
        ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
        return ans
    
    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            bs_mask = (keypoints[:, 0] == k)

            cur_x_idxs = x_idxs[bs_mask]
            cur_y_idxs = y_idxs[bs_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = self.bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
        return point_bev_features
    
    def aggregate_keypoint_features_from_one_source(self, batch_size, aggregate_func, points, new_points, new_points_batch_cnt, filter_neighbors_with_roi = False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None):
        xyz = None
        xyz_features = None
        xyz_batch_cnt = torch.zeros(batch_size).int().to(points[0].device)
        if filter_neighbors_with_roi:
            point_features_list = []
            for bs_idx in range(batch_size):
                xyz = points[bs_idx][:, :3].view(1,-1,3)
                xyz_features = points[bs_idx][:, 3:].view(1, points[bs_idx].shape[0], points[bs_idx].shape[1]-3)
                _, valid_mask = self.sample_points_with_roi(
                    rois=rois[bs_idx], points=xyz[0],
                    sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part,
                )
                point_features_list.append(points[bs_idx][valid_mask])
                xyz_batch_cnt[bs_idx] = valid_mask.sum()
            valid_point_features = torch.cat(point_features_list, dim=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:, 3:]
        pooled_points, pooled_features = aggregate_func(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_points,
            new_xyz_batch_cnt=new_points_batch_cnt,
            features=xyz_features.contiguous(),
        )

        return pooled_features

    def forward(self, points, bboxes, spatial_features, multiscale_feats = None): #multiscale_feats
        print("##### Voxel Set Abstraction #####")
        batch_size = bboxes.shape[0]
        keypoints = self.get_keypoints(points, bboxes, batch_size)
        print("Shape of Keypoints", keypoints.shape)

        point_features_list = []

        #BEV processing
        point_bev_features = self.interpolate_from_bev_features(
            keypoints, spatial_features , batch_size,
                bev_stride= 8
        )

        print("BEV features shape", point_bev_features.shape)

        point_features_list.append(point_bev_features)
        
        new_xyz = keypoints[:, 1:4].contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum()
        
        #Raw point processing
        pooled_features = self.aggregate_keypoint_features_from_one_source(
            batch_size=batch_size, aggregate_func=self.SA_rawpoints,
            points=points,
            new_points=new_xyz, new_points_batch_cnt=new_xyz_batch_cnt,
            filter_neighbors_with_roi=True,
            radius_of_neighbor=2.4,
            rois=bboxes
        )
        point_features_list.append(pooled_features)

        print("Raw point feature shape", pooled_features.shape)

        #x_conv_3
        cur_coords = multiscale_feats["multi_scale_3d_features"]["x_conv3_pos"]
        cur_features = multiscale_feats["multi_scale_3d_features"]["x_conv3"]

        combined_feats = []

        for i in range(len(cur_coords)):
            coords = cur_coords[i]
            coords = self.get_voxel_centers(coords, 4, self.voxel_size, self.point_cloud_range)
            feats = cur_features[i]
            combined = torch.cat([coords, feats], dim = -1)
            combined_feats.append(combined)

        pooled_features = self.aggregate_keypoint_features_from_one_source(
            batch_size=batch_size, aggregate_func=self.SA_layers[0],
            points=combined_feats,
            new_points=new_xyz, new_points_batch_cnt=new_xyz_batch_cnt,
            filter_neighbors_with_roi=True,
            radius_of_neighbor=4.0,
            rois=bboxes
        )
        point_features_list.append(pooled_features)

        print("x_conv3 features shape", pooled_features.shape)

        point_features = torch.cat(point_features_list, dim=-1)

        print("Concatenated feature shape", point_features.shape)

        point_features_before_fusion = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        point_features = point_features  # (BxN, C)
        point_coords = keypoints  # (BxN, 4)
        return point_features, point_coords, point_features_before_fusion

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

class PVRCNNKeypointWeightComputer(nn.Module):
    def __init__(self, num_class, input_channels, cls_fc, gt_extra_width):
        super().__init__()
        self.num_class = num_class
        self.cls_fc = cls_fc
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.cls_fc,
            input_channels=input_channels,
            output_channels=num_class
        )
        self.gt_extra_width = gt_extra_width
    
    def make_fc_layers(self, fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def forward(self, batch_size, point_features, point_coords, point_features_before_fusion, gt_boxes = None, gt_labels = None):
        features_to_use = point_features_before_fusion
        point_cls_preds = self.cls_layers(point_features)
        point_cls_scores = torch.sigmoid(point_cls_preds)

        point_cls_scores, _ = point_cls_scores.max(dim=-1)

        targets_dict = self.assign_targets(batch_size, point_coords, gt_boxes, gt_labels)

        return point_cls_scores

    def enlarge_box3d(self, gt_boxes, extra_width = (0, 0, 0)):
        extend_gt_boxes = []
        for i in range(len(gt_boxes)):
            cur_boxes = gt_boxes[i]
            boxes3d, is_numpy = check_numpy_to_torch(cur_boxes)
            large_boxes3d = boxes3d.clone()

            large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
            extend_gt_boxes.append(large_boxes3d)
        return extend_gt_boxes


    def assign_targets(self, batch_size, points, _gt_boxes, _gt_labels):
        batch_size = len(_gt_boxes)
        gt_boxes = []
        for i in range(batch_size):
            gt_boxes.append(torch.cat([_gt_boxes[i], _gt_labels[i].view(-1,1)], dim = 1))

        extend_gt_boxes = self.enlarge_box3d(gt_boxes, self.gt_extra_width)
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        bs_idx_point = points[:, 0]
        for bs_idx in range(batch_size):
            bs_mask = (bs_idx_point == bs_idx)
            xyz = points[bs_mask][:, 1:4]#points[bs_idx][:, :3].view(-1,3)
            print(xyz.shape)
            point_cls_labels_single = xyz.new_zeros(xyz.shape[0]).long()
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                xyz.unsqueeze(dim=0), gt_boxes[bs_idx][:, 0:7].unsqueeze(dim=0).contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                xyz.unsqueeze(dim=0), extend_gt_boxes[bs_idx][:, 0:7].unsqueeze(dim=0).contiguous()
            ).long().squeeze(dim=0)
            fg_flag = box_fg_flag
            ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
            point_cls_labels_single[ignore_flag] = -1

            gt_box_of_fg_points = gt_boxes[bs_idx][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
        targets_dict = {
            'point_cls_labels': point_cls_labels
        }

        return targets_dict


class PVRCNNPlusPlusProposalTargetLayer(nn.Module):
    #def __init__(self, roi_per_image, fg_ratio, cls_score_type, cls_fg_thresh, cls_bg_thresh, cls_bg_thresh_lo, hard_bg_ratio, reg_fg_thresh):
    def __init__(self, roi_sampler_cfg): 
        super().__init__()

        self.roi_per_image = roi_sampler_cfg.ROI_PER_IMAGE
        self.fg_ratio = roi_sampler_cfg.FG_RATIO
        self.cls_score_type = roi_sampler_cfg.CLS_SCORE_TYPE
        self.cls_fg_thresh = roi_sampler_cfg.CLS_FG_THRESH
        self.cls_bg_thresh = roi_sampler_cfg.CLS_BG_THRESH
        self.cls_bg_thresh_lo = roi_sampler_cfg.CLS_BG_THRESH_LO
        self.hard_bg_ratio = roi_sampler_cfg.HARD_BG_RATIO
        self.reg_fg_thresh = roi_sampler_cfg.REG_FG_THRESH
    
    def forward(self, rois, scores, gt_boxes, roi_labels):
        batch_size = rois.shape[0]
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_size, rois, scores, gt_boxes, roi_labels
        )

        reg_valid_mask = (batch_roi_ious > self.reg_fg_thresh).long()

        iou_bg_thresh = self.cls_bg_thresh
        iou_fg_thresh = self.cls_fg_thresh
        fg_mask = batch_roi_ious > iou_fg_thresh
        bg_mask = batch_roi_ious < iou_bg_thresh
        interval_mask = (fg_mask == 0) & (bg_mask == 0)

        batch_cls_labels = (fg_mask > 0).float()
        batch_cls_labels[interval_mask] = \
            (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        
        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious,
                        'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels,
                        'reg_valid_mask': reg_valid_mask,
                        'rcnn_cls_labels': batch_cls_labels}

        return targets_dict

    def sample_rois_for_rcnn(self, batch_size, rois, scores, gt_boxes, roi_labels):
        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_per_image, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_per_image, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_per_image)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_per_image)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_per_image), dtype=torch.long)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], scores[index]
            k = cur_gt.__len__() - 1
            while k >= 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt

            max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                rois=cur_roi, roi_labels=cur_roi_labels,
                gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
            )

            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels
    
    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds
    
    def subsample_rois(self, max_overlaps):
        # sample fg, easy_bg, hard_bg
        fg_rois_per_image = int(np.round(self.fg_ratio * self.roi_per_image))
        fg_thresh = min(self.reg_fg_thresh, self.cls_fg_thresh)

        fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
        easy_bg_inds = ((max_overlaps < self.cls_bg_thresh_lo)).nonzero().view(-1)
        hard_bg_inds = ((max_overlaps < self.reg_fg_thresh) &
                (max_overlaps >= self.cls_bg_thresh_lo)).nonzero().view(-1)

        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_per_image - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.hard_bg_ratio
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_per_image) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = fg_inds[fg_inds < 0] # yield empty tensor

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_per_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.hard_bg_ratio
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    def iou_3d_compute(self, boxes_a, boxes_b):

        # height overlap
        boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
        boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
        boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
        boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

        boxes_xyxy_a = boxes_a[:,[0,1,3,4]]
        boxes_xyxy_a[:,:2] = boxes_xyxy_a[:,:2] - (boxes_xyxy_a[:,2:]/2)
        boxes_xyxy_a[:,2:] = boxes_xyxy_a[:,2:] + (boxes_xyxy_a[:,:2])
        boxes_xyxy_b = boxes_b[:,[0,1,3,4]]
        boxes_xyxy_b[:,:2] = boxes_xyxy_b[:,:2] - (boxes_xyxy_b[:,2:]/2)
        boxes_xyxy_b[:,2:] = boxes_xyxy_b[:,2:] + (boxes_xyxy_b[:,:2])

        overlaps_bev = bbox_overlaps(boxes_xyxy_a, boxes_xyxy_b)

        max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
        min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
        overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

        # 3d iou
        overlaps_3d = overlaps_bev * overlaps_h
        vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
        vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

        iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

        return iou3d
    
    def get_max_iou_with_same_class(self, rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)

                iou3d = self.iou_3d_compute(cur_roi, cur_gt)  # (M, N)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

        return max_overlaps, gt_assignment
    


"""
Does ROI grid pooling and final refinement step 
"""
class PVRCNNPlusPlusBoxRefinement(nn.Module):
    def __init__(self, input_channels, num_class, code_size) -> None:
        super().__init__()
        config = cfg_from_yaml_file("/gsoc/OpenPCDet/tools/cfgs/waymo_models/pv_rcnn_plusplus.yaml", cfg)
        self.model_cfg = cfg.MODEL.ROI_HEAD
        self.target_config = self.model_cfg.TARGET_CONFIG
        self.roi_grid_pool_config = self.model_cfg.ROI_GRID_POOL
        self.code_size = code_size

        self.num_class = num_class
        self.proposal_target_layer = PVRCNNPlusPlusProposalTargetLayer(roi_sampler_cfg=self.target_config)
        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )
    
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    
    def forward(self, point_coords, point_features, rois, scores, roi_labels, _gt_boxes, _gt_labels, keypoint_weights):
        batch_size = rois.shape[0]
        gt_boxes = []
        for i in range(batch_size):
            gt_boxes.append(torch.cat([_gt_boxes[i], _gt_labels[i].view(-1,1)], dim = 1))
        targets_dict = self.assign_targets(batch_size, rois, scores, gt_boxes, roi_labels)
        rois = targets_dict["rois"]
        roi_labels = targets_dict['roi_labels']

        pooled_features = self.roi_grid_pool(batch_size, rois, point_coords, point_features, keypoint_weights)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        targets_dict['rcnn_cls'] = rcnn_cls
        targets_dict['rcnn_reg'] = rcnn_reg

        return targets_dict


    def assign_targets(self, batch_size, rois, scores, gt_boxes, roi_labels):
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(rois, scores, gt_boxes, roi_labels)
        
        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = self.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def rotate_points_along_z(self, points, angle):
        """
        Args:
            points: (B, N, 3 + C)
            angle: (B), angle along z-axis, angle increases x ==> y
        Returns:

        """
        points, is_numpy = check_numpy_to_torch(points)
        angle, _ = check_numpy_to_torch(angle)

        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot.numpy() if is_numpy else points_rot
    
    def get_dense_grid_points(self, rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = self.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points
    
    def roi_grid_pool(self, batch_size, rois, point_coords, point_features, keypoint_weights):
        point_features = point_features * keypoint_weights.view(-1, 1)
        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.roi_grid_pool_config.GRID_SIZE
        )
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)
        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()
        
        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )

        return pooled_features

class SubmanifoldSparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=0.0,
                 normalize=False):
        super(SubmanifoldSparseConv, self).__init__()
        offset = torch.full((3,), 0.0, dtype=torch.float32)
        self.net = SparseConv(in_channels=in_channels,
                              filters=filters,
                              kernel_size=kernel_size,
                              use_bias=use_bias,
                              offset=offset,
                              normalize=normalize)

    def forward(self,
                features_list,
                in_positions_list,
                out_positions_list=None,
                voxel_size=1.0):
        if out_positions_list is None:
            out_positions_list = in_positions_list

        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list,
                                         out_positions_list):
            out_feat.append(self.net(feat, in_pos.float(), out_pos.float(), voxel_size))

        return out_feat

    def __name__(self):
        return "SubmanifoldSparseConv"

def calculate_grid(in_positions, point_cloud_range, voxel_size, padding, downsample = False):
    dtype = torch.float32
    filter = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                filter.append([i-2, j-2, k-2])
    filter = torch.tensor(np.array(filter)).to(in_positions.device).to(dtype)
    out_pos = in_positions.long().repeat(1, filter.shape[0]).reshape(-1, 3)
    filter = filter.repeat(in_positions.shape[0], 1)

    out_pos = out_pos + filter
    out_pos = out_pos[out_pos.min(1).values >= 0]
    if downsample:
        out_pos = out_pos[(~((out_pos.long() % 2).bool()).any(1))]
    # out_pos = out_pos[out_pos.max(1).values < ((point_cloud_range.max() - point_cloud_range.min())/voxel_size)]
    out_pos = torch.unique(out_pos, dim=0)
    return out_pos

class Convolution(nn.Module):
    """
    SparseConv using a 3x3 kernel
    """
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 point_cloud_range,
                 voxel_size,
                 padding = 0,
                 use_bias=False,
                 offset=None,
                 normalize=False,
                 downsample = False):
        super(Convolution, self).__init__()
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.padding = padding
        self.downsample = downsample
        self.max_num_points = 9
        offset = torch.full((3,), 0.0, dtype=torch.float32)
        self.net = SparseConv(in_channels=in_channels,
                              filters=filters,
                              kernel_size=kernel_size,
                              use_bias=use_bias,
                              offset=offset,
                              normalize=normalize)

    def forward(self, features_list, in_positions_list, voxel_size=1.0):
        out_positions_list = []
        for in_positions in in_positions_list:
            out_positions_list.append(calculate_grid(in_positions, torch.tensor(self.point_cloud_range), voxel_size, self.padding, downsample = self.downsample))
        out_feat = []
        strided_positions = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list,
                                         out_positions_list):
            if self.downsample:
                feat_temp = self.net(feat, in_pos.float(), out_pos.float(), voxel_size).to(feat.device)
                stride_temp = torch.cat([torch.zeros_like(feat_temp[0:1, :]), feat_temp])
                stride_voxelised_2 = voxelize(out_pos.float(), torch.LongTensor([0, out_pos.shape[0]]).to(feat.device), torch.tensor([2.0, 2.0, 2.0]).to(feat.device), torch.Tensor([0, 0, 0]).to(feat.device), torch.Tensor([40960, 40960, 40960]).to(feat.device))
                stride_indices_dense = ragged_to_dense(
                    stride_voxelised_2.voxel_point_indices, stride_voxelised_2.voxel_point_row_splits,
                    self.max_num_points, torch.tensor(-1))+1
                stride_voxels = stride_temp[stride_indices_dense]
                num_points = stride_voxelised_2.voxel_point_row_splits[
                    1:] - stride_voxelised_2.voxel_point_row_splits[:-1]
                voxel_mean = stride_voxels[:, :, :].sum(dim=1, keepdim=False)
                normalizer = torch.clamp_min(num_points.view(-1, 1), min=1.0).type_as(stride_voxels)
                voxel_mean = voxel_mean / normalizer
                out_feat.append(voxel_mean.contiguous())
                stride_positions = torch.tensor(stride_voxelised_2.voxel_coords).to(feat.device).to(torch.float32)
                strided_positions.append(stride_positions)
            else:
                out_feat.append(self.net(feat, in_pos.float(), out_pos.float(), voxel_size))

        # out_positions_list = [out / 2 for out in out_positions_list]
        if self.downsample:
            return out_feat, strided_positions
        return out_feat, out_positions_list

    def __name__(self):
        return "Convolution"

class BatchNormBlock(nn.Module):

    def __init__(self, m, eps=1e-4, momentum=0.01):
        super(BatchNormBlock, self).__init__()
        self.bn = nn.BatchNorm1d(m, eps=eps, momentum=momentum)

    def forward(self, feat_list):
        lengths = [feat.shape[0] for feat in feat_list]
        out = self.bn(torch.cat(feat_list, 0))
        out_list = []
        start = 0
        for l in lengths:
            out_list.append(out[start:start + l])
            start += l

        return out_list

    def __name__(self):
        return "BatchNormBlock"

class ReLUBlock(nn.Module):

    def __init__(self):
        super(ReLUBlock, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, feat_list):
        lengths = [feat.shape[0] for feat in feat_list]
        out = self.relu(torch.cat(feat_list, 0))
        out_list = []
        start = 0
        for l in lengths:
            out_list.append(out[start:start + l])
            start += l

        return out_list

    def __name__(self):
        return "ReLUBlock"

class SparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, poincloud_range, voxel_size, padding = 0, norm_fn = None, downsample = False):
        super(SparseConvBlock, self).__init__()
        self.downsample = downsample
        self.conv = Convolution(in_channels, out_channels, kernel_size, poincloud_range, voxel_size, padding, downsample = downsample)
        self.norm_fn = norm_fn(out_channels)
        self.relu = ReLUBlock()
    
    def forward(self, features_list, in_positions_list, voxel_size = 1.0):
        out_feat, out_positions_list = self.conv(features_list, in_positions_list, voxel_size = voxel_size)
        out_feat = self.norm_fn(out_feat)
        out_feat = self.relu(out_feat)

        return out_feat, out_positions_list

class SubmanifoldSparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_fn = None):
        super(SubmanifoldSparseConvBlock, self).__init__()
        self.conv = SubmanifoldSparseConv(in_channels, out_channels, kernel_size)
        self.norm_fn = norm_fn(out_channels)
        self.relu = ReLUBlock()
    
    def forward(self, features_list, in_positions_list, voxel_size=1.0):
        out_feat = self.conv(features_list, in_positions_list, voxel_size = voxel_size)
        out_feat = self.norm_fn(out_feat)
        out_feat = self.relu(out_feat)

        return out_feat, in_positions_list


MODEL._register_module(PVRCNNPlusPlus, 'torch')