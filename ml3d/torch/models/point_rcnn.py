import torch
from torch import nn

import numpy as np

from .base_model_objdet import BaseModel
from ..modules.losses.focal_loss import FocalLoss
from ..modules.losses.cross_entropy import CrossEntropyLoss
from . import Pointnet2MSG
from ..utils.objdet_helper import xywhr_to_xyxyr, box3d_to_bev
from open3d.ml.torch.ops import nms
from ..utils.torch_utils import gen_CNN

class PointRCNN(BaseModel):
    def __init__(self,
                 name="PointRCNN",
                 device="cuda",
                 use_rpn=True,
                 use_rcnn=True,
                 rpn={},
                 rcnn={},
                 proposal={},
                 **kwargs):
        super().__init__(name=name,
                         device=device,
                         **kwargs)
        self.use_rpn = use_rpn
        self.use_rcnn = use_rcnn

        self.rpn = RPN(**rpn)
        self.rcnn = RCNN(**rcnn)
        self.proposal_layer = ProposalLayer(**proposal)

    def forward(self, inputs):
        if self.use_rpn:
            with torch.set_grad_enabled(self.training):
                output = self.rpn(inputs)
            
            if self.use_rcnn:
                with torch.no_grad():
                    rpn_cls, rpn_reg, backbone_xyz, backbone_features = output

                    rpn_scores_raw = rpn_cls[:, :, 0]
                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                    pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

                    # proposal layer
                    rois, roi_scores_raw = self.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

                    rcnn_output = self.rcnn_net(inputs)

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
        for i in range(len(cls_out_ch)):
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
        for i in range(len(reg_out_ch)):
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
        backbone_xyz, backbone_features = self.backbone_net(x)  # (B, N, 3), (B, C, N)

        rpn_cls = self.cls_blocks(backbone_features).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.reg_blocks(backbone_features).transpose(1, 2).contiguous()  # (B, N, C)

        return rpn_cls, rpn_reg, backbone_xyz, backbone_features

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
                 pool_extra_width=1.0,
                 num_points=512,
                 xyz_up_layer=[512, 512],
                 loss={}):

        super().__init__()
        self.rcnn_input_channel = 5
        
        self.xyz_up_layer = gen_CNN([self.rcnn_input_channel] + xyz_up_layer, conv=nn.Conv2d)
        c_out = xyz_up_layer[-1]
        self.merge_down_layer = gen_CNN([c_out * 2, c_out] + xyz_up_layer, conv=nn.Conv2d)

        self.pool_extra_width = pool_extra_width
        self.num_points = num_points

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
        for i in range(len(cls_out_ch)):
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
        for i in range(len(reg_out_ch)):
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

        self.proposal_target_layer = ProposalTargetLayer(self.pool_extra_width, self.num_points)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, x):
        roi_boxes3d, gt_boxes3d, rpn_xyz, rpn_features, rpn, seg_mask, pts_depth = x
        pts_extra_input_list = [seg_mask.unsqueeze(dim=2)]
        pts_extra_input_list.append((pts_depth / 70.0 - 0.5).unsqueeze(dim=2))
        pts_extra_input = torch.cat(pts_extra_input_list, dim=2)
        pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)

        if self.training:
            with torch.no_grad():
                sampled_pts, pts_feature = self.proposal_target_layer([roi_boxes3d, gt_boxes3d, rpn_xyz, pts_feature])
            pts_input = torch.cat((sampled_pts, pts_feature), dim=2)
        else:
            pooled_features, pooled_empty_flag = roipool3d(rpn_xyz, pts_feature, roi_boxes3d, self.pool_extra_width,
                                                           sampled_pt_num=self.num_points)
                  
            # canonical transformation
            batch_size = roi_boxes3d.shape[0]
            roi_center = roi_boxes3d[:, :, 0:3]
            pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
            for k in range(batch_size):
                pooled_features[k, :, :, 0:3] = rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                        roi_boxes3d[k, :, 6])

            pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])


            xyz, features = self._break_up_pc(pts_input)

            xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
            xyz_feature = self.xyz_up_layer(xyz_input)

            rpn_feature = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)

            merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]

            for i in range(len(self.SA_modules)):
                li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                l_xyz.append(li_xyz)
                l_features.append(li_features)

            rcnn_cls = self.cls_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
            rcnn_reg = self.reg_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

            if self.training:
                return rcnn_cls, rcnn_reg, target
            return rcnn_cls, rcnn_reg

def rotate_pc_along_y(pc, rot_angle):
    """
    params pc: (N, 3+C), (N, 3) is in the rectified camera coordinate
    params rot_angle: rad scalar
    Output pc: updated pc with XYZ rotated
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc

class ProposalLayer(nn.Module):
    def __init__(self, 
                 nms_pre=9000, 
                 nms_post=512):
        super().__init__()
        self.nms_pre = nms_pre
        self.nms_post = nms_post

    def forward(self, x):
        pn_scores, rpn_reg, xyz = x

        batch_size = xyz.shape[0]
        proposals = decode_bbox_target(xyz.view(-1, 3), rpn_reg.view(-1, rpn_reg.shape[-1]),
                                       anchor_size=self.MEAN_SIZE,
                                       loc_scope=cfg.RPN.LOC_SCOPE,
                                       loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                       num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                                       get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                                       get_y_by_bin=False,
                                       get_ry_fine=False)  # (N, 7)
        proposals[:, 1] += proposals[:, 3] / 2  # set y as the center of bottom
        proposals = proposals.view(batch_size, -1, 7)

        scores = rpn_scores
        _, sorted_idxs = torch.sort(scores, dim=1, descending=True)

        batch_size = scores.size(0)
        ret_bbox3d = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N, 7).zero_()
        ret_scores = scores.new(batch_size, cfg[self.mode].RPN_POST_NMS_TOP_N).zero_()
        for k in range(batch_size):
            scores_single = scores[k]
            proposals_single = proposals[k]
            order_single = sorted_idxs[k]

            scores_single, proposals_single = self.distance_based_proposal(scores_single, proposals_single,
                                                                               order_single)

            proposals_tot = proposals_single.size(0)
            ret_bbox3d[k, :proposals_tot] = proposals_single
            ret_scores[k, :proposals_tot] = scores_single

        return ret_bbox3d, ret_scores

    def distance_based_proposal(self, scores, proposals, order):
        """
         propose rois in two area based on the distance
        :param scores: (N)
        :param proposals: (N, 7)
        :param order: (N)
        """
        nms_range_list = [0, 40.0, 80.0]
        pre_top_n_list = [0, int(self.nms_pre * 0.7), self.nms_pre - int(self.nms_pre * 0.7)]
        post_top_n_list = [0, int(self.nms_post * 0.7), self.nms_post - int(self.nms_post * 0.7)]

        scores_single_list, proposals_single_list = [], []

        # sort by score
        scores_ordered = scores[order]
        proposals_ordered = proposals[order]

        dist = proposals_ordered[:, 2]
        first_mask = (dist > nms_range_list[0]) & (dist <= nms_range_list[1])
        for i in range(1, len(nms_range_list)):
            # get proposal distance mask
            dist_mask = ((dist > nms_range_list[i - 1]) & (dist <= nms_range_list[i]))

            if dist_mask.sum() != 0:
                # this area has points
                # reduce by mask
                cur_scores = scores_ordered[dist_mask]
                cur_proposals = proposals_ordered[dist_mask]

                # fetch pre nms top K
                cur_scores = cur_scores[:pre_top_n_list[i]]
                cur_proposals = cur_proposals[:pre_top_n_list[i]]
            else:
                assert i == 2, '%d' % i
                # this area doesn't have any points, so use rois of first area
                cur_scores = scores_ordered[first_mask]
                cur_proposals = proposals_ordered[first_mask]

                # fetch top K of first area
                cur_scores = cur_scores[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]
                cur_proposals = cur_proposals[pre_top_n_list[i - 1]:][:pre_top_n_list[i]]

            # oriented nms
            bev = xywhr_to_xyxyr(box3d_to_bev(cur_proposals))
            keep_idx = nms(bev, cur_scores, 0.8)

            # Fetch post nms top k
            keep_idx = keep_idx[:post_top_n_list[i]]

            scores_single_list.append(cur_scores[keep_idx])
            proposals_single_list.append(cur_proposals[keep_idx])

        scores_single = torch.cat(scores_single_list, dim=0)
        proposals_single = torch.cat(proposals_single_list, dim=0)
        return scores_single, proposals_single


def rotate_pc_along_y_torch(pc, rot_angle):
    """
    :param pc: (N, 3 + C)
    :param rot_angle: (N)
    :return:
    """
    cosa = torch.cos(rot_angle).view(-1, 1)
    sina = torch.sin(rot_angle).view(-1, 1)

    raw_1 = torch.cat([cosa, -sina], dim=1)
    raw_2 = torch.cat([sina, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)  # (N, 2, 2)

    pc_temp = pc[:, [0, 2]].unsqueeze(dim=1)  # (N, 1, 2)

    pc[:, [0, 2]] = torch.matmul(pc_temp, R.permute(0, 2, 1)).squeeze(dim=1)
    return pc

def decode_bbox_target(roi_box3d, pred_reg, loc_scope, loc_bin_size, num_head_bin, anchor_size)
    """
    :param roi_box3d: (N, 7)
    :param pred_reg: (N, C)
    :param loc_scope:
    :param loc_bin_size:
    :param num_head_bin:
    :param anchor_size:
    :param get_xz_fine:
    :return:
    """
    anchor_size = anchor_size.to(roi_box3d.get_device())
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2

    # recover xz localization
    x_bin_l, x_bin_r = 0, per_loc_bin_num
    z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    start_offset = z_bin_r

    x_bin = torch.argmax(pred_reg[:, x_bin_l: x_bin_r], dim=1)
    z_bin = torch.argmax(pred_reg[:, z_bin_l: z_bin_r], dim=1)

    pos_x = x_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
    pos_z = z_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope

    x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
    z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
    start_offset = z_res_r

    x_res_norm = torch.gather(pred_reg[:, x_res_l: x_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
    z_res_norm = torch.gather(pred_reg[:, z_res_l: z_res_r], dim=1, index=z_bin.unsqueeze(dim=1)).squeeze(dim=1)
    x_res = x_res_norm * loc_bin_size
    z_res = z_res_norm * loc_bin_size

        pos_x += x_res
        pos_z += z_res

    # recover y localization
    y_offset_l, y_offset_r = start_offset, start_offset + 1
    start_offset = y_offset_r

    pos_y = roi_box3d[:, 1] + pred_reg[:, y_offset_l]

    # recover ry rotation
    ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
    ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

    ry_bin = torch.argmax(pred_reg[:, ry_bin_l: ry_bin_r], dim=1)
    ry_res_norm = torch.gather(pred_reg[:, ry_res_l: ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)

    angle_per_class = (2 * np.pi) / num_head_bin
    ry_res = ry_res_norm * (angle_per_class / 2)

    # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
    ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
    ry[ry > np.pi] -= 2 * np.pi

    # recover size
    size_res_l, size_res_r = ry_res_r, ry_res_r + 3
    assert size_res_r == pred_reg.shape[1]

    size_res_norm = pred_reg[:, size_res_l: size_res_r]
    hwl = size_res_norm * anchor_size + anchor_size

    # shift to original coords
    roi_center = roi_box3d[:, 0:3]
    shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_y.view(-1, 1), pos_z.view(-1, 1), hwl, ry.view(-1, 1)), dim=1)
    ret_box3d = shift_ret_box3d
    if roi_box3d.shape[1] == 7:
        roi_ry = roi_box3d[:, 6]
        ret_box3d = rotate_pc_along_y_torch(shift_ret_box3d, - roi_ry)
        ret_box3d[:, 6] += roi_ry
    ret_box3d[:, [0, 2]] += roi_center[:, [0, 2]]

    return ret_box3d


# TODO:
class ProposalTargetLayer(nn.Module):
    def __init__(self,
            pool_extra_width=1.0,
            num_points=512,
            reg_fg_thresh,
            reg_bg_thresh,
            cls_fg_thresh,
            cls_bg_thresh,
            cls_bg_thresh_lo,
            fg_ratio,
            roi_per_image,
            CLS_BG_THRESH_LO
            ):
        super().__init__()
        self.pool_extra_width = pool_extra_width
        self.num_points = num_points

    def forward(self, x):
        roi_boxes3d, gt_boxes3d, rpn_xyz, pts_feature = x
        batch_rois, batch_gt_of_rois, batch_roi_iou = self.sample_rois_for_rcnn(roi_boxes3d, gt_boxes3d)

        # point cloud pooling
        pooled_features, pooled_empty_flag = \
            roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, self.pool_extra_width,
                                          sampled_pt_num=self.num_points)

        sampled_pts, sampled_features = pooled_features[:, :, :, 0:3], pooled_features[:, :, :, 3:]

        # canonical transformation
        batch_size = batch_rois.shape[0]
        roi_ry = batch_rois[:, :, 6] % (2 * np.pi)
        roi_center = batch_rois[:, :, 0:3]
        sampled_pts = sampled_pts - roi_center.unsqueeze(dim=2)  # (B, M, 512, 3)
        batch_gt_of_rois[:, :, 0:3] = batch_gt_of_rois[:, :, 0:3] - roi_center
        batch_gt_of_rois[:, :, 6] = batch_gt_of_rois[:, :, 6] - roi_ry

        for k in range(batch_size):
            sampled_pts[k] = rotate_pc_along_y_torch(sampled_pts[k], batch_rois[k, :, 6])
            batch_gt_of_rois[k] = rotate_pc_along_y_torch(batch_gt_of_rois[k].unsqueeze(dim=1),
                                                          roi_ry[k]).squeeze(dim=1)

        # regression valid mask
        valid_mask = (pooled_empty_flag == 0)
        reg_valid_mask = ((batch_roi_iou > cfg.RCNN.REG_FG_THRESH) & valid_mask).long()

        # classification label
        batch_cls_label = (batch_roi_iou > cfg.RCNN.CLS_FG_THRESH).long()
        invalid_mask = (batch_roi_iou > cfg.RCNN.CLS_BG_THRESH) & (batch_roi_iou < cfg.RCNN.CLS_FG_THRESH)
        batch_cls_label[valid_mask == 0] = -1
        batch_cls_label[invalid_mask > 0] = -1

        output_dict = {'sampled_pts': sampled_pts.view(-1, cfg.RCNN.NUM_POINTS, 3),
                       'pts_feature': sampled_features.view(-1, cfg.RCNN.NUM_POINTS, sampled_features.shape[3]),
                       'cls_label': batch_cls_label.view(-1),
                       'reg_valid_mask': reg_valid_mask.view(-1),
                       'gt_of_rois': batch_gt_of_rois.view(-1, 7),
                       'gt_iou': batch_roi_iou.view(-1),
                       'roi_boxes3d': batch_rois.view(-1, 7)}

        return output_dict

    def sample_rois_for_rcnn(self, roi_boxes3d, gt_boxes3d):
        """
        :param roi_boxes3d: (B, M, 7)
        :param gt_boxes3d: (B, N, 8) [x, y, z, h, w, l, ry, cls]
        :return
            batch_rois: (B, N, 7)
            batch_gt_of_rois: (B, N, 8)
            batch_roi_iou: (B, N)
        """
        batch_size = roi_boxes3d.size(0)

        fg_rois_per_image = int(np.round(cfg.RCNN.FG_RATIO * cfg.RCNN.ROI_PER_IMAGE))

        batch_rois = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE, 7).zero_()
        batch_gt_of_rois = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE, 7).zero_()
        batch_roi_iou = gt_boxes3d.new(batch_size, cfg.RCNN.ROI_PER_IMAGE).zero_()

        for idx in range(batch_size):
            cur_roi, cur_gt = roi_boxes3d[idx], gt_boxes3d[idx]

            k = cur_gt.__len__() - 1
            while cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]

            # include gt boxes in the candidate rois
            iou3d = iou3d_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (M, N)

            max_overlaps, gt_assignment = torch.max(iou3d, dim=1)

            # sample fg, easy_bg, hard_bg
            fg_thresh = min(cfg.RCNN.REG_FG_THRESH, cfg.RCNN.CLS_FG_THRESH)
            fg_inds = torch.nonzero((max_overlaps >= fg_thresh)).view(-1)

            # TODO: this will mix the fg and bg when CLS_BG_THRESH_LO < iou < CLS_BG_THRESH
            # fg_inds = torch.cat((fg_inds, roi_assignment), dim=0)  # consider the roi which has max_iou with gt as fg

            easy_bg_inds = torch.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH_LO)).view(-1)
            hard_bg_inds = torch.nonzero((max_overlaps < cfg.RCNN.CLS_BG_THRESH) &
                                         (max_overlaps >= cfg.RCNN.CLS_BG_THRESH_LO)).view(-1)

            fg_num_rois = fg_inds.numel()
            bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes3d).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE - fg_rois_per_this_image
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                rand_num = np.floor(np.random.rand(cfg.RCNN.ROI_PER_IMAGE) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes3d).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                bg_rois_per_this_image = cfg.RCNN.ROI_PER_IMAGE
                bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image)

                fg_rois_per_this_image = 0
            else:
                import pdb
                pdb.set_trace()
                raise NotImplementedError

            # augment the rois by noise
            roi_list, roi_iou_list, roi_gt_list = [], [], []
            if fg_rois_per_this_image > 0:
                fg_rois_src = cur_roi[fg_inds]
                gt_of_fg_rois = cur_gt[gt_assignment[fg_inds]]
                iou3d_src = max_overlaps[fg_inds]
                fg_rois, fg_iou3d = self.aug_roi_by_noise_torch(fg_rois_src, gt_of_fg_rois, iou3d_src,
                                                                aug_times=cfg.RCNN.ROI_FG_AUG_TIMES)
                roi_list.append(fg_rois)
                roi_iou_list.append(fg_iou3d)
                roi_gt_list.append(gt_of_fg_rois)

            if bg_rois_per_this_image > 0:
                bg_rois_src = cur_roi[bg_inds]
                gt_of_bg_rois = cur_gt[gt_assignment[bg_inds]]
                iou3d_src = max_overlaps[bg_inds]
                aug_times = 1 if cfg.RCNN.ROI_FG_AUG_TIMES > 0 else 0
                bg_rois, bg_iou3d = self.aug_roi_by_noise_torch(bg_rois_src, gt_of_bg_rois, iou3d_src,
                                                                aug_times=aug_times)
                roi_list.append(bg_rois)
                roi_iou_list.append(bg_iou3d)
                roi_gt_list.append(gt_of_bg_rois)

            rois = torch.cat(roi_list, dim=0)
            iou_of_rois = torch.cat(roi_iou_list, dim=0)
            gt_of_rois = torch.cat(roi_gt_list, dim=0)

            batch_rois[idx] = rois
            batch_gt_of_rois[idx] = gt_of_rois
            batch_roi_iou[idx] = iou_of_rois

        return batch_rois, batch_gt_of_rois, batch_roi_iou

    def sample_bg_inds(self, hard_bg_inds, easy_bg_inds, bg_rois_per_this_image):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = int(bg_rois_per_this_image * cfg.RCNN.HARD_BG_RATIO)
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds