import numpy as np
import torch
import torch.nn as nn

from .base_model import BaseModel
from ...utils import MODEL
from ..modules.losses import filter_valid_label
from ...datasets.augment import SemsegAugmentation
from open3d.ml.torch.layers import SparseConv, SparseConvTranspose
from open3d.ml.torch.ops import voxelize, reduce_subarrays_sum


class SparseConvUnet(BaseModel):

    def __init__(
            self,
            name="SparseConvUnet",
            device="cuda",
            m=16,
            voxel_size=0.05,
            reps=1,  # Conv block repetitions.
            residual_blocks=False,
            in_channels=3,
            num_classes=20,
            **kwargs):
        super(SparseConvUnet, self).__init__(name=name,
                                             device=device,
                                             m=m,
                                             voxel_size=voxel_size,
                                             reps=reps,
                                             residual_blocks=residual_blocks,
                                             in_channels=in_channels,
                                             num_classes=num_classes,
                                             **kwargs)
        cfg = self.cfg
        self.device = device
        self.augment = SemsegAugmentation(cfg.augment)
        self.m = cfg.m
        self.inp = InputLayer()
        self.ssc = SubmanifoldSparseConv(in_channels=in_channels,
                                         filters=m,
                                         kernel_size=[3, 3, 3])
        self.unet = UNet(reps, [m, 2 * m, 3 * m, 4 * m, 5 * m, 6 * m, 7 * m],
                         residual_blocks)
        self.bn = BatchNormBlock(m)
        self.relu = ReLUBlock()
        self.linear = LinearBlock(m, num_classes)
        self.out = OutputLayer()

    def forward(self, inputs):
        pos_list = []
        feat_list = []
        rev_list = []

        for i in range(len(inputs.batch_lengths)):
            pos = inputs.point[i]
            feat = inputs.feat[i]
            feat, pos, rev = self.inp(feat, pos)
            pos_list.append(pos)
            feat_list.append(feat)
            rev_list.append(rev)

        feat_list = self.ssc(feat_list, pos_list, voxel_size=1.0)
        feat_list = self.unet(pos_list, feat_list)
        feat_list = self.bn(feat_list)
        feat_list = self.relu(feat_list)
        feat_list = self.linear(feat_list)
        output = self.out(feat_list, rev_list)

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

        feat = np.array(data['feat'], dtype=np.float32)

        # Scale to voxel size.
        points *= 1. / self.cfg.voxel_size  # Scale = 1/voxel_size

        if attr['split'] in ['training', 'train']:
            points, feat, labels = self.augment.augment(
                points, feat, labels, self.cfg.get('augment', None))

        feat = feat / 127.5 - 1.

        m = points.min(0)
        M = points.max(0)
        offset = -m + np.clip(4096 - M + m - 0.001, 0, None) * np.random.rand(
            3) + np.clip(4096 - M + m + 0.001, None, 0) * np.random.rand(3)

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
        data['point'] = torch.from_numpy(data['point'])
        data['feat'] = torch.from_numpy(data['feat'])
        data['label'] = torch.from_numpy(data['label'])

        return data

    def inference_begin(self, data):
        data = self.preprocess(data, {'split': 'test'})
        data['batch_lengths'] = [data['point'].shape[0]]
        data = self.transform(data, {})

        self.inference_input = data

    def inference_preprocess(self):
        return self.inference_input

    def inference_end(self, inputs, results):
        results = torch.reshape(results, (-1, self.cfg.num_classes))

        m_softmax = torch.nn.Softmax(dim=-1)
        results = m_softmax(results)
        results = results.cpu().data.numpy()

        probs = np.reshape(results, [-1, self.cfg.num_classes])

        pred_l = np.argmax(probs, 1)

        return {'inference_labels': pred_l, 'inference_scores': probs}

    def get_loss(self, Loss, results, inputs, device):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """
        cfg = self.cfg
        labels = torch.cat(inputs['data'].label, 0)

        scores, labels = filter_valid_label(results, labels, cfg.num_classes,
                                            cfg.ignored_label_inds, device)

        loss = Loss.weighted_CrossEntropyLoss(scores, labels)

        return loss, labels, scores

    def get_optimizer(self, cfg_pipeline):
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg_pipeline.adam_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, cfg_pipeline.scheduler_gamma)

        return optimizer, scheduler


MODEL._register_module(SparseConvUnet, 'torch')


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


class LinearBlock(nn.Module):

    def __init__(self, a, b):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(a, b)

    def forward(self, feat_list):
        out_list = []
        for feat in feat_list:
            out_list.append(self.linear(feat))

        return out_list

    def __name__(self):
        return "LinearBlock"


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

    def forward(self, features_list, rev_list):
        out = []
        for feat, rev in zip(features_list, rev_list):
            out.append(feat[rev])
        return torch.cat(out, 0)


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
                features_list,
                inp_positions_list,
                out_positions_list=None,
                voxel_size=1.0):
        if out_positions_list is None:
            out_positions_list = inp_positions_list

        out_feat = []
        for feat, inp_pos, out_pos in zip(features_list, inp_positions_list,
                                          out_positions_list):
            out_feat.append(self.net(feat, inp_pos, out_pos, voxel_size))

        return out_feat

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

    def forward(self, features_list, inp_positions_list, voxel_size=1.0):
        out_positions_list = []
        for inp_positions in inp_positions_list:
            out_positions_list.append(calculate_grid(inp_positions))

        out_feat = []
        for feat, inp_pos, out_pos in zip(features_list, inp_positions_list,
                                          out_positions_list):
            out_feat.append(self.net(feat, inp_pos, out_pos, voxel_size))

        out_positions_list = [out / 2 for out in out_positions_list]

        return out_feat, out_positions_list

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

    def forward(self,
                features_list,
                inp_positions_list,
                out_positions_list,
                voxel_size=1.0):
        out_feat = []
        for feat, inp_pos, out_pos in zip(features_list, inp_positions_list,
                                          out_positions_list):
            out_feat.append(self.net(feat, inp_pos, out_pos, voxel_size))

        return out_feat

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
        out = []
        for a, b in zip(feat_cat, feat):
            out.append(torch.cat([a, b], -1))

        return out


class NetworkInNetwork(nn.Module):

    def __init__(self, nIn, nOut, bias=False):
        super(NetworkInNetwork, self).__init__()
        if nIn == nOut:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(nIn, nOut, bias=bias)

    def forward(self, inputs):
        out = []
        for inp in inputs:
            out.append(self.linear(inp))

        return out


class ResidualBlock(nn.Module):

    def __init__(self, nIn, nOut):
        super(ResidualBlock, self).__init__()

        self.lin = NetworkInNetwork(nIn, nOut)

        self.bn1 = BatchNormBlock(nIn)
        self.relu1 = ReLUBlock()
        self.scn1 = SubmanifoldSparseConv(in_channels=nIn,
                                          filters=nOut,
                                          kernel_size=[3, 3, 3])

        self.bn2 = BatchNormBlock(nOut)
        self.relu2 = ReLUBlock()
        self.scn2 = SubmanifoldSparseConv(in_channels=nOut,
                                          filters=nOut,
                                          kernel_size=[3, 3, 3])

    def forward(self, feat_list, pos_list):
        out1 = self.lin(feat_list)

        feat_list = self.bn1(feat_list)

        feat_list = self.relu1(feat_list)

        feat_list = self.scn1(feat_list, pos_list)

        feat_list = self.bn2(feat_list)
        feat_list = self.relu2(feat_list)

        out2 = self.scn2(feat_list, pos_list)

        return [a + b for a, b in zip(out1, out2)]

    def __name__(self):
        return "ResidualBlock"


class UNet(nn.Module):

    def __init__(self,
                 reps,
                 nPlanes,
                 residual_blocks=False,
                 downsample=[2, 2],
                 leakiness=0):
        super(UNet, self).__init__()
        self.net = nn.ModuleList(self.U(nPlanes, residual_blocks, reps))
        self.residual_blocks = residual_blocks

    @staticmethod
    def block(m, a, b, residual_blocks):
        if residual_blocks:
            m.append(ResidualBlock(a, b))

        else:
            m.append(BatchNormBlock(a))
            m.append(ReLUBlock())
            m.append(
                SubmanifoldSparseConv(in_channels=a,
                                      filters=b,
                                      kernel_size=[3, 3, 3]))

    @staticmethod
    def U(nPlanes, residual_blocks, reps):
        m = []
        for i in range(reps):
            UNet.block(m, nPlanes[0], nPlanes[0], residual_blocks)

        if len(nPlanes) > 1:
            m.append(ConcatFeat())
            m.append(BatchNormBlock(nPlanes[0]))
            m.append(ReLUBlock())
            m.append(
                Convolution(in_channels=nPlanes[0],
                            filters=nPlanes[1],
                            kernel_size=[2, 2, 2]))
            m = m + UNet.U(nPlanes[1:], residual_blocks, reps)
            m.append(BatchNormBlock(nPlanes[1]))
            m.append(ReLUBlock())
            m.append(
                DeConvolution(in_channels=nPlanes[1],
                              filters=nPlanes[0],
                              kernel_size=[2, 2, 2]))

            m.append(JoinFeat())

            for i in range(reps):
                UNet.block(m, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0],
                           residual_blocks)

        return m

    def forward(self, pos_list, feat_list):
        conv_pos = []
        concat_feat = []
        for module in self.net:
            if isinstance(module, BatchNormBlock):
                feat_list = module(feat_list)
            elif isinstance(module, ReLUBlock):
                feat_list = module(feat_list)

            elif module.__name__() == "ResidualBlock":
                feat_list = module(feat_list, pos_list)

            elif module.__name__() == "SubmanifoldSparseConv":
                feat_list = module(feat_list, pos_list)

            elif module.__name__() == "Convolution":
                conv_pos.append([pos.clone() for pos in pos_list])
                feat_list, pos_list = module(feat_list, pos_list)

            elif module.__name__() == "DeConvolution":
                feat_list = module(feat_list, [2 * pos for pos in pos_list],
                                   conv_pos[-1])
                pos_list = conv_pos.pop()

            elif module.__name__() == "ConcatFeat":
                concat_feat.append([feat.clone() for feat in module(feat_list)])

            elif module.__name__() == "JoinFeat":
                feat_list = module(concat_feat.pop(), feat_list)

            else:
                raise Exception("Unknown module {}".format(module))

        return feat_list


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
