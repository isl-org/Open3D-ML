import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

import open3d

if open3d.core.cuda.device_count() > 0:
    from open3d.ml.torch.ops import furthest_point_sampling, gather_points, gather_points_grad, three_nn, three_interpolate, three_interpolate_grad, group_points, group_points_grad, ball_query


class FurthestPointSampling(Function):

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:tensor containing the set
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        assert xyz.is_contiguous()
        output = furthest_point_sampling(xyz, npoint)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        assert features.is_contiguous()
        assert idx.is_contiguous()

        _, C, N = features.size()
        output = gather_points(features, idx)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        idx, C, N = ctx.for_backwards
        grad_out_data = grad_out.data.contiguous()
        grad_features = gather_points_grad(grad_out_data, idx, C, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, query_pts: torch.Tensor,
                data_pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of query_pts in data_pts
        :param ctx:
        :param query_pts: (B, N, 3)
        :param data_pts: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        assert query_pts.is_contiguous()
        assert data_pts.is_contiguous()

        dist2, idx = three_nn(query_pts, data_pts)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn_gpu = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        """
        Performs weight linear interpolation on 3 features
        :param ctx:
        :param features: (B, C, M) Features descriptors to be interpolated from
        :param idx: (B, n, 3) three nearest neighbors of the target features in features
        :param weight: (B, n, 3) weights
        :return:
            output: (B, C, N) tensor of the interpolated features
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        ctx.three_interpolate_for_backward = (idx, weight, features.size()[2])
        output = three_interpolate(features, idx, weight)
        return output

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, N) tensor with gradients of outputs
        :return:
            grad_features: (B, C, M) tensor with gradients of features
            None:
            None:
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        idx, weight, m = ctx.three_interpolate_for_backward

        grad_out_data = grad_out.data.contiguous()
        grad_features = three_interpolate_grad(grad_out_data, idx, weight, m)
        return grad_features, None, None


three_interpolate_gpu = ThreeInterpolate.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        assert features.is_contiguous()
        assert idx.is_contiguous()

        output = group_points(features, idx)

        ctx.for_backwards = (idx, features.size()[2])
        return output

    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        idx, N = ctx.for_backwards

        grad_out_data = grad_out.data.contiguous()
        grad_features = group_points_grad(grad_out_data, idx, N)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor,
                new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        idx = ball_query(xyz, new_xyz, radius, nsample)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query_gpu = BallQuery.apply


class QueryAndGroup(nn.Module):

    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        """
        :param radius: float, radius of ball
        :param nsample: int, maximum number of features to gather in the ball
        :param use_xyz:
        """
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self,
                xyz: torch.Tensor,
                new_xyz: torch.Tensor,
                features: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        idx = ball_query_gpu(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans,
                                         idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self,
                xyz: torch.Tensor,
                new_xyz: torch.Tensor,
                features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
