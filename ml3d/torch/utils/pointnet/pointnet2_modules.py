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
#***************************************************************************************/s

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils
from . import pytorch_utils as pt_utils
from typing import List


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def forward(self,
                xyz: torch.Tensor,
                features: torch.Tensor = None,
                new_xyz=None) -> (torch.Tensor, torch.Tensor):
        r"""Forward.

        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        if new_xyz is None and self.npoint is not None:
            sampling = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            new_xyz = torch.gather(xyz, 1,
                                   torch.stack([sampling] * 3, -1).long())

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz,
                                            features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](
                new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features,
                                            kernel_size=[
                                                1, new_features.size(3)
                                            ])  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features,
                                            kernel_size=[
                                                1, new_features.size(3)
                                            ])  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping."""

    def __init__(self,
                 *,
                 npoint: int,
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 instance_norm=False):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(
                pt_utils.SharedMLP(mlp_spec, bn=bn,
                                   instance_norm=instance_norm))
        self.pool_method = pool_method


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer."""

    def __init__(self,
                 *,
                 mlp: List[int],
                 npoint: int = None,
                 radius: float = None,
                 nsample: int = None,
                 bn: bool = True,
                 use_xyz: bool = True,
                 pool_method='max_pool',
                 instance_norm=False):
        """PointnetSAModule.

        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param instance_norm: whether to use instance_norm
        """
        super().__init__(mlps=[mlp],
                         npoint=npoint,
                         radii=[radius],
                         nsamples=[nsample],
                         bn=bn,
                         use_xyz=use_xyz,
                         pool_method=pool_method,
                         instance_norm=instance_norm)


class PointnetFPModule(nn.Module):
    r"""Propagates the features of one set to another."""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor,
                unknow_feats: torch.Tensor,
                known_feats: torch.Tensor) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn_gpu(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate_gpu(
                known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2],
                                                    unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats],
                                     dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    pass
