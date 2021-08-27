import torch
from torch.autograd import Function
import torch.nn as nn
import open3d

if open3d.core.cuda.device_count() > 0:
    from open3d.ml.torch.ops import trilinear_devoxelize_forward, three_interpolate_for_backward


class TrilinearDevoxelization(Function):

    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """
        :param ctx:
        :param coords: the coordinates of points, FloatTensor[B, 3, N]
        :param features: FloatTensor[B, C, R, R, R]
        :param resolution: int, the voxel resolution
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        B, C = features.shape[:2]
        features = features.contiguous().view(B, C, -1)
        coords = coords.contiguous()
        outs, inds, wgts = trilinear_devoxelize_forward(resolution, is_training,
                                                        coords, features)
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: 
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, R, R, R]
        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = trilinear_devoxelize_backward(grad_output.contiguous(),
                                                    inds, wgts, ctx.r)
        return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.r,
                                ctx.r, ctx.r), None, None, None


trilinear_devoxelize = TrilinearDevoxelization.apply
