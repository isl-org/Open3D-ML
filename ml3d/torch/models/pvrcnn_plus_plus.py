import torch
from torch import nn
from torch.nn.modules.utils import _pair

from .base_model_objdet import BaseModel
import open3d.ml.torch as ml3d
from open3d.ml.torch.ops import voxelize, ragged_to_dense, reduce_subarrays_sum



class PVRCNNPlusPlus(BaseModel):
    def __init__(self, 
                 name = "PVRCNN++",
                 device="cuda",
                 point_cloud_range=[0, -40.0, -3, 70.0, 40.0, 1],
                 classes=['car'],
                 **kwargs):
        super().__init__(name=name,
                         point_cloud_range=point_cloud_range,
                         device=device,
                         **kwargs)
        self.point_cloud_range = point_cloud_range
        self.classes = classes
        self.device = device
        self.to(device)

    @torch.no_grad()
    def voxelize(self, points):
        """Apply voxelization to points."""
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
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, inputs):
        pass

"""
Convert the Last layer of 3D Sparse Convolution network to Birds Eye-View via Height compression
"""
class PVRCNNPlusPlusBEVModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, feats):
        pass

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



        