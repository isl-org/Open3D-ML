from numpy import pad
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torch.nn import functional as F
from functools import partial
import numpy as np

from .base_model_objdet import BaseModel
import open3d.ml.torch as ml3d
from open3d.ml.torch.ops import voxelize, ragged_to_dense, reduce_subarrays_sum
from open3d.ml.torch.layers import SparseConv



class PVRCNNPlusPlus(BaseModel):
    def __init__(self, 
                 name = "PVRCNN++",
                 device="cuda",
                 point_cloud_range=[0, -40.0, -3, 70.0, 40.0, 1],
                 classes=['car'],
                 voxelize={},
                 voxel_encoder={},
                 **kwargs):
        super().__init__(name=name,
                         point_cloud_range=point_cloud_range,
                         device=device,
                         **kwargs)
        self.point_cloud_range = point_cloud_range
        self.classes = classes
        self.device = device
        self.to(device)
        self.voxel_layer = PVRCNNPlusPlusVoxelization(point_cloud_range=point_cloud_range, **voxelize)
        self.sparseconvbackbone = PVRCNNPlusPlusBackbone3D(point_cloud_range=point_cloud_range, **voxel_encoder)

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

        return voxels, voxel_features, num_points, coors_batch
    
    def convert_open3d_voxel_to_dense(self, x_open3d, x_pos_open3d, voxel_size = 1.0):
        x_sparse = []
        dense_shape = ((self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(voxel_size)).tolist()
        dense_shape = dense_shape.append(x.shape[1])
        for x, x_pos in zip(x_open3d, x_pos_open3d):
            x_sparse_temp = torch.sparse_coo_tensor(torch.tensor(x_pos).t(), x, dense_shape)
            x_sparse.append(x_sparse_temp)
        return x_sparse

    def backbone_3d(self, points):
        voxels, voxel_features, num_points, coors = self.voxelize(points)
        x_intermediate_layers, x, x_pos = self.sparseconvbackbone(voxel_features, coors)  
        return x, x_intermediate_layers, x_pos
    
    def backbone_2d(self, input_channels, layer_nums, layer_strides, num_filters, num_upsample_filters, upsample_strides):
        



    def forward(self, inputs):
        inputs = inputs.point
        x_dense_bev, x, x_intermediate_layers, x_pos = self.backbone_3d(inputs)
        x_sparse = self.convert_open3d_voxel_to_dense(x, x_pos)
        x_dense_bev = self.bev_module(x_sparse)





    
    def get_optimizer(self, cfg):
        pass

    def get_loss(self, results, inputs):
        pass

    def preprocess(self, data, attr):
        pass

    def transform(self, data, attr):
        pass

    def inference_end(self, results, inputs):
        pass

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

"""
Layer to obtain outputs from different layers of 3D Sparse Convolution Network
Will also need to store the intermediate layer outputs
"""
class PVRCNNPlusPlusBackbone3D(nn.Module):
    def __init__(self, in_channels, poincloud_range, voxel_size):
        super().__init__()
        self.point_cloud_range = poincloud_range
        self.voxel_size = voxel_size
        norm_fn = norm_fn = partial(BatchNormBlock, eps=1e-3, momentum=0.01)

        self.conv_input = SubmanifoldSparseConvBlock(in_channels, 16, 3, norm_fn = norm_fn)

        self.conv1 = SparseConvBlock(16, 16, 3, poincloud_range, voxel_size, padding=1, norm_fn=norm_fn)
        
        self.conv2 = nn.Sequential(
            SparseConvBlock(16, 32, 3, poincloud_range, voxel_size, padding=1, norm_fn=norm_fn), #stride=2
            SparseConvBlock(32, 32, 3, poincloud_range, voxel_size, padding=1, norm_fn=norm_fn),
            SparseConvBlock(32, 32, 3, poincloud_range, voxel_size, padding=1, norm_fn=norm_fn),
        )

        self.conv3 = nn.Sequential(
            SparseConvBlock(32, 64, 3, poincloud_range, voxel_size, padding=1, norm_fn=norm_fn), #stride=2
            SparseConvBlock(64, 64, 3, poincloud_range, voxel_size, padding=1, norm_fn=norm_fn),
            SparseConvBlock(64, 64, 3, poincloud_range, voxel_size, padding=1, norm_fn=norm_fn),
        )

        self.conv4 = nn.Sequential(
            SparseConvBlock(64, 64, 3, poincloud_range, voxel_size, padding=1, norm_fn=norm_fn), #stride=2, padding = (0, 1, 1)
            SparseConvBlock(64, 64, 3, poincloud_range, voxel_size, padding=1, norm_fn=norm_fn),
            SparseConvBlock(64, 64, 3, poincloud_range, voxel_size, padding=1, norm_fn=norm_fn),
        )

        self.conv_out = SparseConvBlock(64, 128, 3, poincloud_range, voxel_size, padding = 0, norm_fn = norm_fn) #stride = (2, 1, 1)


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
    
    def forward(self, sparse_tensor_list):
        x_dense_list = []
        for x_sparse in sparse_tensor_list:
            spatial_features = x_sparse.dense().permute(3, 0, 1, 2)
            C, D, H, W= spatial_features.shape
            spatial_features = spatial_features.view(C * D, H, W)
            x_dense_list.append(spatial_features)
        x_dense = torch.cat(x_dense_list, dim=0)
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

"""
Proposes and classifies 3D Bounding Boxes
one head for the following : center - 2, center_z - 1, dim - 3, rot - 2 and class
Applies NMS here as well
Outputs rois, roi_scores, roi_labels, has_class_labels
"""
class RPNModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, bev_feats):
        pass

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
    def __init__(self) -> None:
        super().__init__()

    def get_keypoints(self, boxes, points):
        pass

    def forward(self, points, boxes, multiscale_feats, bev_feats):
        pass

"""
Does ROI grid pooling and final refinement step 
"""
class PVRCNNPlusPlusBoxRefinement(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def roi_grid_pooling(self, roi, points, point_coords, point_feats, keypoint_weights):
        pass


class SubmanifoldSparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=None,
                 normalize=False):
        super(SubmanifoldSparseConv, self).__init__()

        offset = torch.full((3,), offset, dtype=torch.float32)
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
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))

        return out_feat

    def __name__(self):
        return "SubmanifoldSparseConv"

def calculate_grid(in_positions, point_cloud_range, voxel_size, padding):
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
    out_pos = out_pos[out_pos.max(1).values < ((point_cloud_range.max() - point_cloud_range.min())/voxel_size)]
    out_pos = torch.unique(out_pos, dim=0)

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
                 normalize=False):
        super(Convolution, self).__init__()
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.padding = padding
        self.net = SparseConv(in_channels=in_channels,
                              filters=filters,
                              kernel_size=kernel_size,
                              use_bias=use_bias,
                              #offset=offset,
                              normalize=normalize)

    def forward(self, features_list, in_positions_list, voxel_size=1.0):
        out_positions_list = []
        for in_positions in in_positions_list:
            out_positions_list.append(calculate_grid(in_positions, self.point_cloud_range, self.voxel_size, self.padding))

        out_feat = []
        for feat, in_pos, out_pos in zip(features_list, in_positions_list,
                                         out_positions_list):
            out_feat.append(self.net(feat, in_pos, out_pos, voxel_size))

        # out_positions_list = [out / 2 for out in out_positions_list]

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
    def __init__(self, in_channels, out_channels, kernel_size, poincloud_range, voxel_size, padding = 0, norm_fn = None):
        super(SparseConvBlock, self).__init__()
        self.conv = Convolution(in_channels, out_channels, kernel_size, poincloud_range, voxel_size, padding)
        self.norm_fn = norm_fn(out_channels)
        self.relu = ReLUBlock()
    
    def forward(self, inputs):
        features_list, in_positions_list, voxel_size = inputs
        out_feat, out_positions_list = self.conv(features_list, in_positions_list, voxel_size)
        out_feat = self.norm_fn(out_feat)
        out_feat = self.relu(out_feat)

        return (out_feat, out_positions_list, voxel_size)

class SubmanifoldSparseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm_fn = None):
        super(SubmanifoldSparseConvBlock, self).__init__()
        self.conv = SubmanifoldSparseConv(in_channels, out_channels, kernel_size)
        self.norm_fn = norm_fn(out_channels)
        self.relu = ReLUBlock()
    
    def forward(self, features_list, in_positions_list, voxel_size=1.0):
        out_feat = self.conv(features_list, in_positions_list, voxel_size)
        out_feat = self.norm_fn(out_feat)
        out_feat = self.relu(out_feat)

        return (out_feat, in_positions_list, voxel_size)
        