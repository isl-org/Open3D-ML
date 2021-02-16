import numpy as np
import torch
import torch.nn as nn

from .base_model import BaseModel
from ...utils import MODEL

import open3d.ml.torch as ml3d


class SparseConvUnet(BaseModel):

    def __init__(self,
                 name="SparseConvUnet",
                 device="cuda",
                 m=16,
                 in_channels=3,
                 num_classes=20,
                 **kwargs):
        super(SparseConvUnet, self).__init__(name=name,
                                             device=device,
                                             m=m,
                                             in_channels=in_channels,
                                             num_classes=num_classes,
                                             **kwargs)
        cfg = self.cfg
        self.m = cfg.m
        self.ssc = SubmanifoldSparseConv(in_channels=in_channels,
                                         filters=m,
                                         kernel_size=[3, 3, 3])
        self.unet = UNet(1, [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m],
                         False)
        self.bn = nn.BatchNorm1d(m, eps=1e-4, momentum=0.01)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(m, num_classes)

    def forward(self, inputs):
        output = []
        for inp in inputs:
            pos = inp['point']
            feat = inp['feat']

            feat = self.ssc(feat, pos, voxel_size=1.0)
            feat = self.unet(pos, feat)
            feat = self.bn(feat)
            feat = self.relu(feat)
            feat = self.linear(feat)

            output.append(feat)
        return output

    def preprocess(self, data, attr):
        cfg = self.cfg

        points = np.array(data['point'], dtype=np.float32)

        if 'label' not in data.keys() or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data.keys() or data['feat'] is None:
            raise Exception(
                "SparseConvnet doesn't work without feature values.")

        feat = np.array(data['feat'], dtype=np.float32) / 127.5 - 1

        offset = np.clip(4096 - points.max(0) + points.min(0) - 0.001, 0,
                         None) / 2

        points += offset
        idxs = (points.min(1) >= 0) * (points.max(1) < 4096)

        points = points[idxs]
        feat = feat[idxs]
        labels = labels[idxs]

        points = (points.astype(np.int32) + 0.5).astype(
            np.float32)  # Move points to voxel center.

        # Take one point from each voxel. Ideally we want to take average of features from same voxel.
        _, index, inv = np.unique(points,
                                  return_index=True,
                                  return_inverse=True,
                                  axis=0)
        index = np.sort(index)

        data = {}
        data['point'] = points[index]
        data['feat'] = feat[index]
        data['label'] = labels[index]
        data['proj_inds'] = inv

        return data

    def transform(self, data, attr):
        device = self.device
        data['point'] = torch.from_numpy(data['point']).to(device)
        data['feat'] = torch.from_numpy(data['feat']).to(device)
        data['label'] = torch.from_numpy(data['label']).to(device)

        return data

    def inference_begin(self, data):
        data = self.preprocess(data, {})
        data = self.transform(data, {})

        self.inference_input = data

    def inference_preprocess(self):
        return [self.inference_input]

    def inference_end(self, inputs, results):
        inputs = inputs[0]
        results = results[0]
        results = torch.reshape(results, (-1, self.cfg.num_classes))
        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()
        probs = np.reshape(results,
                           [-1, self.cfg.num_classes])[inputs['proj_inds']]

        pred_l = np.argmax(probs, 1)
        print(pred_l[:30])

        return {'inference_labels': pred_l, 'inference_scores': probs}

    def get_loss():
        raise NotImplementedError

    def get_optimizer():
        raise NotImplementedError


MODEL._register_module(SparseConvUnet, 'torch')


class SubmanifoldSparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=None,
                 normalize=False):
        super(SubmanifoldSparseConv, self).__init__()

        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.
            else:
                offset = 0.5

        offset = torch.full((3,), offset, dtype=torch.float32)
        self.net = ml3d.layers.SparseConv(in_channels=in_channels,
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          use_bias=use_bias,
                                          offset=offset,
                                          normalize=normalize)

    def forward(self, features, inp_positions, voxel_size=1.0):
        out = self.net(features, inp_positions, inp_positions, voxel_size)
        return out

    def __name__(self):
        return "SubmanifoldSparseConv"


def calculate_grid(inp_positions):
    inp_pos = inp_positions.long()
    out_pos = []
    for p in inp_pos:
        for i in range(-1, 1):
            for j in range(-1, 1):
                for k in range(-1, 1):
                    arr = p + torch.Tensor([i, j, k]).to(p.device)
                    if not torch.any(arr < 0):
                        if arr[0] % 2 or arr[1] % 2 or arr[2] % 2:
                            continue
                        out_pos.append(arr)
    out_pos = torch.cat(out_pos).float().reshape(-1, 3)
    out_pos = torch.unique(out_pos, dim=0)

    return out_pos + 0.5


class Convolution(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=None,
                 normalize=False):
        super(Convolution, self).__init__()

        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.
            else:
                offset = -0.5

        offset = torch.full((3,), offset, dtype=torch.float32)
        self.net = ml3d.layers.SparseConv(in_channels=in_channels,
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          use_bias=use_bias,
                                          offset=offset,
                                          normalize=normalize)

    def forward(self, features, inp_positions, voxel_size=1.0):
        out_positions = calculate_grid(inp_positions)
        out = self.net(features, inp_positions, out_positions, voxel_size)
        return out, out_positions / 2

    def __name__(self):
        return "Convolution"


class DeConvolution(nn.Module):

    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 use_bias=False,
                 offset=None,
                 normalize=False):
        super(DeConvolution, self).__init__()

        if offset is None:
            if kernel_size[0] % 2:
                offset = 0.
            else:
                offset = -0.5

        offset = torch.full((3,), offset, dtype=torch.float32)
        self.net = ml3d.layers.SparseConvTranspose(in_channels=in_channels,
                                                   filters=filters,
                                                   kernel_size=kernel_size,
                                                   use_bias=use_bias,
                                                   offset=offset,
                                                   normalize=normalize)

    def forward(self, features, inp_positions, out_positions, voxel_size=1.0):
        return self.net(features, inp_positions, out_positions, voxel_size)

    def __name__(self):
        return "DeConvolution"


class ConcatFeat(nn.Module):

    def __init__(self):
        super(ConcatFeat, self).__init__()

    def __name__(self):
        return "ConcatFeat"

    def forward(self, feat):
        return feat


class JoinFeat(nn.Module):

    def __init__(self):
        super(JoinFeat, self).__init__()

    def __name__(self):
        return "JoinFeat"

    def forward(self, feat_cat, feat):
        return torch.cat([feat_cat, feat], -1)


class UNet(nn.Module):

    def __init__(self,
                 reps,
                 nPlanes,
                 resudual_blocks=False,
                 downsample=[2, 2],
                 leakiness=0):
        super(UNet, self).__init__()
        self.net = nn.ModuleList(self.U(nPlanes))
        # print(self.net)

    @staticmethod
    def block(m, a, b):
        m.append(nn.BatchNorm1d(a, eps=1e-4, momentum=0.01))
        m.append(nn.LeakyReLU(0))
        m.append(
            SubmanifoldSparseConv(in_channels=a,
                                  filters=b,
                                  kernel_size=[3, 3, 3]))

    @staticmethod
    def U(nPlanes):
        m = []
        UNet.block(m, nPlanes[0], nPlanes[0])

        if len(nPlanes) > 1:
            m.append(ConcatFeat())
            m.append(nn.BatchNorm1d(nPlanes[0], eps=1e-4, momentum=0.01))
            m.append(nn.LeakyReLU(0))
            m.append(
                Convolution(in_channels=nPlanes[0],
                            filters=nPlanes[1],
                            kernel_size=[2, 2, 2]))
            m = m + UNet.U(nPlanes[1:])
            m.append(nn.BatchNorm1d(nPlanes[1], eps=1e-4, momentum=0.01))
            m.append(nn.LeakyReLU(0))
            m.append(
                DeConvolution(in_channels=nPlanes[1],
                              filters=nPlanes[0],
                              kernel_size=[2, 2, 2]))

            m.append(JoinFeat())

            UNet.block(m, 2 * nPlanes[0], nPlanes[0])

        return m

    def forward(self, pos, feat):
        conv_pos = []
        concat_feat = []
        for module in self.net:
            if isinstance(module, nn.BatchNorm1d):
                feat = module(feat)
            elif isinstance(module, nn.LeakyReLU):
                feat = module(feat)

            elif module.__name__() == "SubmanifoldSparseConv":
                feat = module(feat, pos)

            elif module.__name__() == "Convolution":
                conv_pos.append(pos.clone())
                feat, pos = module(feat, pos)
            elif module.__name__() == "DeConvolution":
                feat = module(feat, 2 * pos, conv_pos[-1])
                pos = conv_pos.pop()

            elif module.__name__() == "ConcatFeat":
                concat_feat.append(module(feat).clone())
            elif module.__name__() == "JoinFeat":
                feat = module(concat_feat.pop(), feat)

            else:
                raise Exception("Unknown module {}".format(module))

        return feat


def load_unet_wts(net, path):
    wts = list(torch.load(path).values())
    state_dict = net.state_dict()
    i = 0
    for key in state_dict.keys():
        if 'offset' in key or 'tracked' in key:
            continue
        if len(wts[i].shape) == 4:
            shp = wts[i].shape
            state_dict[key] = np.transpose(
                wts[i].reshape(int(shp[0]**(1 / 3)), int(shp[0]**(1 / 3)),
                               int(shp[0]**(1 / 3)), shp[-2], shp[-1]),
                (2, 1, 0, 3, 4))
        else:
            state_dict[key] = wts[i]
        i += 1

    net.load_state_dict(state_dict)
