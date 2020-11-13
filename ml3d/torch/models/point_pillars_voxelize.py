import torch
import open3d.ml.torch as ml3d
from torch.nn.modules.utils import _pair


class PointPillarsVoxelization(torch.nn.Module):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
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

        ans = ml3d.ops.voxelize(points, self.voxel_size, self.points_range_min,
                                self.points_range_max, self.max_num_points,
                                max_voxels)

        # prepend row with zeros which maps to index 0 which maps to void points.
        feats = torch.cat(
            [torch.zeros_like(points_feats[0:1, :]), points_feats])

        # create dense matrix of indices. index 0 maps to the zero vector.
        voxels_point_indices_dense = ml3d.ops.ragged_to_dense(
            ans.voxel_point_indices, ans.voxel_point_row_splits,
            self.max_num_points, torch.tensor(-1)) + 1

        out_voxels = feats[voxels_point_indices_dense]
        out_coords = ans.voxel_coords[:, [2, 1, 0]].contiguous()
        out_num_points = ans.voxel_point_row_splits[
            1:] - ans.voxel_point_row_splits[:-1]

        return out_voxels, out_coords, out_num_points
