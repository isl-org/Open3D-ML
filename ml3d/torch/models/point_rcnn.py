import torch
from torch import nn

import numpy as np

from .base_model_objdet import BaseModel
from ..modules.losses.focal_loss import FocalLoss
from ..modules.losses.cross_entropy import CrossEntropyLoss

class PointRCNN(BaseModel):
    def __init__(self,
                 name="PointRCNN",
                 device="cuda",
                 use_rpn=True,
                 use_rcnn=True,
                 rpn={},
                 rcnn={},
                 **kwargs):
        super().__init__(name=name,
                         device=device,
                         **kwargs)
        self.use_rpn = use_rpn
        self.use_rcnn = use_rcnn

        self.rpn = RPN(**rpn)
        self.rcnn = RCNN(**rcnn)

    def forward(self, inputs):
        if self.use_rpn:
            with torch.set_grad_enabled(self.training):
                output = self.rpn(inputs)
            
            if self.use_rcnn:
                raise NotImplementedError

        elif self.use_rcnn:
            output = self.rcnn(inputs)

        return output

    def get_optimizer(self, cfg):
        raise NotImplementedError


    def loss(self, results, inputs):
        raise NotImplementedError


    def preprocess(self, data, attr):
        raise NotImplementedError


    def transform(self, data, attr):
        raise NotImplementedError


    def inference_end(self, results, inputs):
        raise NotImplementedError

class RPN(nn.Module):
    def __init__(self,
                 backbone={},
                 cls_in_ch=128,
                 cls_out_ch=[128],
                 reg_in_ch=128,
                 reg_out_ch=[128],
                 db_ratio=0.5,
                 loc_xz_fine=True,
                 loc_scope=3.0,
                 loc_bin_size=0.5,
                 num_head_bin=12,
                 focal_loss={}):

        super().__init__()

        # backbone
        self.backbone = Pointnet2MSG(**backbone)

        # classification branch
        in_filters = [cls_in_ch, *cls_out_ch[:-1]]
        layers = []
        for i in enumerate(len(cls_out_ch)):
            layers.extend([
                nn.Conv1d(in_filters[i],
                          cls_out_ch[i],
                          1, bias=False),
                nn.BatchNorm1d(cls_out_ch[i]),  
                nn.ReLU(inplace=True),
                nn.Dropout(db_ratio)        
            ])
        layers.append(
            nn.Conv1d(cls_out_ch[-1],
                      1, 1, bias=True))

        self.cls_blocks = nn.Sequential(*layers)

        # regression branch
        per_loc_bin_num = loc_scope // loc_bin_size * 2
        if loc_xz_fine:
            reg_channel = per_loc_bin_num * 4 + num_head_bin * 2 + 3
        else:
            reg_channel = per_loc_bin_num * 2 + num_head_bin * 2 + 3
        reg_channel += 1  # reg y

        in_filters = [reg_in_ch, *reg_out_ch[:-1]]
        layers = []
        for i in enumerate(len(reg_out_ch)):
            layers.extend([
                nn.Conv1d(in_filters[i],
                          reg_out_ch[i],
                          1, bias=False),
                nn.BatchNorm1d(reg_out_ch[i]),  
                nn.ReLU(inplace=True),
                nn.Dropout(db_ratio)        
            ])
        layers.append(
            nn.Conv1d(reg_out_ch[-1],
                      reg_channel,
                      1, bias=True))

        self.reg_blocks = nn.Sequential(*layers)

        self.loss_cls = FocalLoss(**focal_loss)

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.cls_blocks[2].conv.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.reg_blocks[-1].conv.weight, mean=0, std=0.001)

    def forward(self, x):
        _, backbone_features = self.backbone_net(x)  # (B, N, 3), (B, C, N)

        rpn_cls = self.cls_blocks(backbone_features).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.reg_blocks(backbone_features).transpose(1, 2).contiguous()  # (B, N, C)

        return rpn_cls, rpn_reg

    def loss(self, results, inputs):
        raise NotImplementedError


class RCNN(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels=0,
                 SA_config={
                    "npoints": [128, 32, -1],
                    "radius": [0.2, 0.4, 100],
                    "nsample": [64, 64, 64],
                    "mlps": [[128, 128, 128],
                             [128, 128, 256],
                             [256, 256, 512]]
                 },
                 cls_out_ch=[128],
                 reg_out_ch=[128],
                 db_ratio=0.5,
                 loc_scope=3.0,
                 loc_bin_size=0.5,
                 num_head_bin=12,
                 loc_y_by_bin=False,
                 loc_y_scope=0.5,
                 loc_y_bin_size=0.25,
                 use_xyz=True,
                 loss={}):

        super().__init__()
        self.rcnn_input_channel = 5
        self.xyz_up_layer = ...
        self.merge_down_layer = ...

        self.SA_modules = nn.ModuleList()
        for i in range(len(SA_config["npoints"])):
            mlps = [in_channels] + SA_config["mlps"][i]
            npoint = SA_config["npoints"][i] if SA_config["npoints"][i] != -1 else None
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=npoint,
                    radius=SA_config["radius"][i],
                    nsample=SA_config["nsample"][i],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=False
                )
            )
            in_channels = mlps[-1]

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes

        in_filters = [in_channels, *cls_out_ch[:-1]]
        layers = []
        for i in enumerate(len(cls_out_ch)):
            layers.extend([
                nn.Conv1d(in_filters[i],
                          cls_out_ch[i],
                          1, bias=True),
                nn.ReLU(inplace=True)       
            ])
        layers.append(
            nn.Conv1d(cls_out_ch[-1],
                      cls_channel, 
                      1, bias=True))

        self.cls_blocks = nn.Sequential(*layers)

        self.loss_cls = nn.functional.binary_cross_entropy

        # regression branch
        per_loc_bin_num = loc_scope // loc_bin_size * 2
        loc_y_bin_num = loc_y_scope // loc_y_bin_size * 2
        reg_channel = per_loc_bin_num * 4 + num_head_bin * 2 + 3
        reg_channel += (1 if not loc_y_by_bin else loc_y_bin_num * 2)
        reg_channel += 1  # reg y

        in_filters = [in_channels, *reg_out_ch[:-1]]
        layers = []
        for i in enumerate(len(reg_out_ch)):
            layers.extend([
                nn.Conv1d(in_filters[i],
                          reg_out_ch[i],
                          1, bias=True),
                nn.ReLU(inplace=True)     
            ])
        layers.append(
            nn.Conv1d(reg_out_ch[-1],
                      reg_channel, 
                      1, bias=True))

        self.reg_blocks = nn.Sequential(*layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)


class Pointnet2MSG(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class PointnetSAModule(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

class ProposalTargetLayer(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError