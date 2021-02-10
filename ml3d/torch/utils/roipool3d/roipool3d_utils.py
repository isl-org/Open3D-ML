import torch
from open3d.ml.torch.ops import roipool3d
import numpy as np


def enlarge_box3d(boxes3d, extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 1] += extra_width
    return large_boxes3d


def roipool3d_gpu(pts,
                  pts_feature,
                  boxes3d,
                  pool_extra_width,
                  sampled_pt_num=512):
    """
    :param pts: (B, N, 3)
    :param pts_feature: (B, N, C)
    :param boxes3d: (B, M, 7)
    :param pool_extra_width: float
    :param sampled_pt_num: int
    :return:
        pooled_features: (B, M, 512, 3 + C)
        pooled_empty_flag: (B, M)
    """
    batch_size = pts.shape[0]
    pooled_boxes3d = enlarge_box3d(boxes3d.view(-1, 7),
                                   pool_extra_width).view(batch_size, -1, 7)

    pooled_features, pooled_empty_flag = roipool3d(pts.contiguous(),
                                                   pooled_boxes3d.contiguous(),
                                                   pts_feature.contiguous(),
                                                   sampled_pt_num)

    return pooled_features, pooled_empty_flag


def pts_in_boxes3d_cpu(pts, boxes3d):
    """
    :param pts: (N, 3) in rect-camera coords
    :param boxes3d: (M, 7)
    :return: boxes_pts_mask_list: (M), list with [(N), (N), ..]
    """
    if not pts.is_cuda:
        pts = pts.float().contiguous()
        boxes3d = boxes3d.float().contiguous()
        pts_flag = torch.LongTensor(torch.Size(
            (boxes3d.size(0), pts.size(0))))  # (M, N)
        roipool3d_cuda.pts_in_boxes3d_cpu(pts_flag, pts, boxes3d)

        boxes_pts_mask_list = []
        for k in range(0, boxes3d.shape[0]):
            cur_mask = pts_flag[k] > 0
            boxes_pts_mask_list.append(cur_mask)
        return boxes_pts_mask_list
    else:
        raise NotImplementedError


def roipool_pc_cpu(pts, pts_feature, boxes3d, sampled_pt_num):
    """
    :param pts: (N, 3)
    :param pts_feature: (N, C)
    :param boxes3d: (M, 7)
    :param sampled_pt_num: int
    :return:
    """
    pts = pts.cpu().float().contiguous()
    pts_feature = pts_feature.cpu().float().contiguous()
    boxes3d = boxes3d.cpu().float().contiguous()
    assert pts.shape[0] == pts_feature.shape[0] and pts.shape[
        1] == 3, '%s %s' % (pts.shape, pts_feature.shape)
    assert pts.is_cuda is False
    pooled_pts = torch.FloatTensor(
        torch.Size((boxes3d.shape[0], sampled_pt_num, 3))).zero_()
    pooled_features = torch.FloatTensor(
        torch.Size(
            (boxes3d.shape[0], sampled_pt_num, pts_feature.shape[1]))).zero_()
    pooled_empty_flag = torch.LongTensor(boxes3d.shape[0]).zero_()
    roipool3d_cuda.roipool3d_cpu(pts, boxes3d, pts_feature, pooled_pts,
                                 pooled_features, pooled_empty_flag)
    return pooled_pts, pooled_features, pooled_empty_flag


def roipool3d_cpu(boxes3d,
                  pts,
                  pts_feature,
                  pts_extra_input,
                  pool_extra_width,
                  sampled_pt_num=512,
                  canonical_transform=True):
    """
    :param boxes3d: (N, 7)
    :param pts: (N, 3)
    :param pts_feature: (N, C)
    :param pts_extra_input: (N, C2)
    :param pool_extra_width: constant
    :param sampled_pt_num: constant
    :return:
    """
    pooled_boxes3d = enlarge_box3d(boxes3d, pool_extra_width)

    pts_feature_all = np.concatenate((pts_extra_input, pts_feature), axis=1)

    #  Note: if pooled_empty_flag[i] > 0, the pooled_pts[i], pooled_features[i] will be zero
    pooled_pts, pooled_features, pooled_empty_flag = \
        roipool_pc_cpu(torch.from_numpy(pts), torch.from_numpy(pts_feature_all),
                       torch.from_numpy(pooled_boxes3d), sampled_pt_num)

    extra_input_len = pts_extra_input.shape[1]
    sampled_pts_input = torch.cat(
        (pooled_pts, pooled_features[:, :, 0:extra_input_len]), dim=2).numpy()
    sampled_pts_feature = pooled_features[:, :, extra_input_len:].numpy()

    if canonical_transform:
        # Translate to the roi coordinates
        roi_ry = boxes3d[:, 6] % (2 * np.pi)  # 0~2pi
        roi_center = boxes3d[:, 0:3]

        # shift to center
        sampled_pts_input[:, :,
                          0:3] = sampled_pts_input[:, :,
                                                   0:3] - roi_center[:, np.
                                                                     newaxis, :]
        for k in range(sampled_pts_input.shape[0]):
            sampled_pts_input[k] = rotate_pc_along_y(sampled_pts_input[k],
                                                     roi_ry[k])

        return sampled_pts_input, sampled_pts_feature

    return sampled_pts_input, sampled_pts_feature, pooled_empty_flag.numpy()


if __name__ == '__main__':
    pass
