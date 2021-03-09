import numpy as np
import torch
import torch.nn as nn

from .base_model import BaseModel
from ...utils import MODEL

from open3d.ml.torch.layers import SparseConv, SparseConvTranspose
from open3d.ml.torch.ops import voxelize, reduce_subarrays_sum


class SparseConvUnet(BaseModel):

    def __init__(self,
                 name="SparseConvUnet",
                 device="cuda",
                 m=16,
                 scale=20,
                 in_channels=3,
                 num_classes=20,
                 **kwargs):
        super(SparseConvUnet, self).__init__(name=name,
                                             device=device,
                                             m=m,
                                             scale=scale,
                                             in_channels=in_channels,
                                             num_classes=num_classes,
                                             **kwargs)
        cfg = self.cfg
        self.device = device
        self.m = cfg.m
        self.inp = InputLayer()
        self.ssc = SubmanifoldSparseConv(in_channels=in_channels,
                                         filters=m,
                                         kernel_size=[3, 3, 3])
        self.unet = UNet(1, [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m],
                         False)
        self.bn = nn.BatchNorm1d(m, eps=1e-4, momentum=0.01)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(m, num_classes)
        self.out = OutputLayer()

    def forward(self, inputs):
        output = []
        for inp in inputs:
            pos = inp['point']
            feat = inp['feat']

            feat, pos, rev = self.inp(feat, pos)
            feat = self.ssc(feat, pos, voxel_size=1.0)
            feat = self.unet(pos, feat)
            feat = self.bn(feat)
            feat = self.relu(feat)
            feat = self.linear(feat)
            feat = self.out(feat, rev)

            output.append(feat)
        return output

    def preprocess(self, data, attr):
        points = np.array(data['point'], dtype=np.float32)

        if 'label' not in data.keys() or data['label'] is None:
            labels = np.zeros((points.shape[0],), dtype=np.int32)
        else:
            labels = np.array(data['label'], dtype=np.int32).reshape((-1,))

        if 'feat' not in data.keys() or data['feat'] is None:
            raise Exception(
                "SparseConvnet doesn't work without feature values.")

        feat = np.array(data['feat'], dtype=np.float32) / 127.5 - 1

        points *= self.cfg.scale  # Scale = 1/voxel_size

        offset = np.clip(4096 - points.max(0) + points.min(0) - 0.001, 0,
                         None) / 2

        points += offset
        idxs = (points.min(1) >= 0) * (points.max(1) < 4096)

        points = points[idxs]
        feat = feat[idxs]
        labels = labels[idxs]

        points = (points.astype(np.int32) + 0.5).astype(
            np.float32)  # Move points to voxel center.

        data = {}
        data['point'] = points
        data['feat'] = feat
        data['label'] = labels

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

        probs = np.reshape(results, [-1, self.cfg.num_classes])

        pred_l = np.argmax(probs, 1)

        return {'inference_labels': pred_l, 'inference_scores': probs}

    def get_loss(self):
        raise NotImplementedError

    def get_optimizer(self):
        raise NotImplementedError


MODEL._register_module(SparseConvUnet, 'torch')


class InputLayer(nn.Module):

    def __init__(self, voxel_size=1.0):
        super(InputLayer, self).__init__()
        self.voxel_size = torch.Tensor([voxel_size, voxel_size, voxel_size])

    def forward(self, features, inp_positions):
        v = voxelize(inp_positions, self.voxel_size, torch.Tensor([0, 0, 0]),
                     torch.Tensor([40960, 40960, 40960]))

        # Contiguous repeating positions.
        inp_positions = inp_positions[v.voxel_point_indices]
        features = features[v.voxel_point_indices]

        # Find reverse mapping.
        rev1 = np.zeros((inp_positions.shape[0],))
        rev1[v.voxel_point_indices.cpu().numpy()] = np.arange(
            inp_positions.shape[0])
        rev1 = rev1.astype(np.int32)

        # Unique positions.
        inp_positions = inp_positions[v.voxel_point_row_splits[:-1]]

        # Mean of features.
        count = v.voxel_point_row_splits[1:] - v.voxel_point_row_splits[:-1]
        rev2 = np.repeat(np.arange(count.shape[0]),
                         count.cpu().numpy()).astype(np.int32)

        features_avg = inp_positions.clone()
        features_avg[:, 0] = reduce_subarrays_sum(features[:, 0],
                                                  v.voxel_point_row_splits)
        features_avg[:, 1] = reduce_subarrays_sum(features[:, 1],
                                                  v.voxel_point_row_splits)
        features_avg[:, 2] = reduce_subarrays_sum(features[:, 2],
                                                  v.voxel_point_row_splits)

        features_avg = features_avg / count.unsqueeze(1)

        return features_avg, inp_positions, rev2[rev1]


class OutputLayer(nn.Module):

    def __init__(self, voxel_size=1.0):
        super(OutputLayer, self).__init__()

    def forward(self, features, rev):
        return features[rev]


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
        self.net = SparseConv(in_channels=in_channels,
                              filters=filters,
                              kernel_size=kernel_size,
                              use_bias=use_bias,
                              offset=offset,
                              normalize=normalize)

    def forward(self,
                features,
                inp_positions,
                out_positions=None,
                voxel_size=1.0):
        if out_positions is None:
            out_positions = inp_positions
        return self.net(features, inp_positions, out_positions, voxel_size)

    def __name__(self):
        return "SubmanifoldSparseConv"


def calculate_grid(inp_positions):
    filter = torch.Tensor([[-1, -1, -1], [-1, -1, 0], [-1, 0, -1], [-1, 0, 0],
                           [0, -1, -1], [0, -1, 0], [0, 0, -1],
                           [0, 0, 0]]).to(inp_positions.device)

    out_pos = inp_positions.long().repeat(1, filter.shape[0]).reshape(-1, 3)
    filter = filter.repeat(inp_positions.shape[0], 1)

    out_pos = out_pos + filter
    out_pos = out_pos[out_pos.min(1).values >= 0]
    out_pos = out_pos[(~((out_pos.long() % 2).bool()).any(1))]
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
        self.net = SparseConv(in_channels=in_channels,
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
        self.net = SparseConvTranspose(in_channels=in_channels,
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
