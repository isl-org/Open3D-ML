#***************************************************************************************/
#
#    Based on Pointnet2 Library (MIT license):
#    https://github.com/sshaoshuai/Pointnet2.PyTorch
#
#    Copyright (c) 2019 Shaoshuai Shi

#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:

#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.
#
#***************************************************************************************/

import torch
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

import open3d

if open3d.core.cuda.device_count() > 0:
    from open3d.ml.torch.ops import furthest_point_sampling, three_nn, three_interpolate, three_interpolate_grad, ball_query


class FurthestPointSampling(Function):

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """Uses iterative furthest point sampling to select a set of npoint
        features that have the largest minimum distance.

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


class FurthestPointSamplingV2(Function):
    """Furthest Point Sampling with variable length batch support."""

    @staticmethod
    def forward(ctx, xyz, row_splits, new_row_splits):
        """Forward pass.

        Args:
            ctx: Context.
            xyz (torch.float32): Input pointcloud (N, 3).
            row_splits (torch,int64): splits to define batch (b + 1,)
            new_row_splits (torch.int64): splits for output batch lengths (b + 1,)

        Returns:
            Returns indices of sampled points with shape (new_row_splits[-1], ).
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        if not xyz.is_contiguous():
            raise ValueError(
                "FurthestPointSampling : coordinates are not contiguous.")

        idx = []
        for i in range(0, row_splits.shape[0] - 1):
            npoint = new_row_splits[i + 1] - new_row_splits[i]
            start_i = row_splits[i]
            end_i = row_splits[i + 1]
            out = furthest_point_sampling(xyz[start_i:end_i].unsqueeze(0),
                                          npoint) + row_splits[i]

            idx += out

        return torch.cat(idx, 0)

    @staticmethod
    def backward(xyz, a=None, b=None):
        return None, None, None


furthest_point_sample_v2 = FurthestPointSamplingV2.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, query_pts: torch.Tensor,
                data_pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the three nearest neighbors of query_pts in data_pts.

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
        """Performs weight linear interpolation on 3 features.

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
        """Backward.

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


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor,
                new_xyz: torch.Tensor) -> torch.Tensor:
        """Forward.

        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indices of the features that form the query balls
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
        """QueryAndGroup.

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
        """Forward.

        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        if not open3d.core.cuda.device_count() > 0:
            raise NotImplementedError

        batch_size = xyz.shape[0]

        idx = ball_query_gpu(self.radius, self.nsample, xyz, new_xyz)
        idx_stacked = torch.stack([idx] * 3, dim=1).view(batch_size, 3,
                                                         -1).long()

        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = torch.gather(xyz_trans, dim=2, index=idx_stacked).view(
            batch_size, 3, -1, self.nsample)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            idx_stacked = torch.stack([idx] * features.shape[1],
                                      dim=1).view(batch_size, features.shape[1],
                                                  -1).long()
            grouped_features = torch.gather(features, dim=2,
                                            index=idx_stacked).view(
                                                batch_size, features.shape[1],
                                                -1, self.nsample)
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
        """Forward.

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
