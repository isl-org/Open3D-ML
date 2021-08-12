#***************************************************************************************/
#
#    Based on MMDetection3D Library (Apache 2.0 license):
#    https://github.com/open-mmlab/mmdetection3d
#
#    Copyright 2018-2019 Open-MMLab.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#***************************************************************************************/

import numpy as np
import torch

from functools import partial

from open3d.ml.torch.ops import nms


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int,
                           device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        torch.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period


def xywhr_to_xyxyr(boxes_xywhr):
    """Convert rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns:
        torch.Tensor: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


def box3d_to_bev(boxes3d):
    """Convert rotated 3d boxes in XYZWHDR format to BEV in XYWHR format.

    Args:
        boxes3d (torch.Tensor): Rotated boxes in XYZWHDR format.

    Returns:
        torch.Tensor: Converted BEV boxes in XYWHR format.
    """
    return boxes3d[:, [0, 1, 3, 4, 6]]


def box3d_to_bev2d(boxes3d):
    """Convert rotated 3d boxes in XYZWHDR format to neareset BEV without
    rotation.

    Args:
        boxes3d (torch.Tensor): Rotated boxes in XYZWHDR format.

    Returns:
        torch.Tensor: Converted BEV boxes in XYWH format.
    """
    # Obtain BEV boxes with rotation in XYWHR format
    bev_rotated_boxes = box3d_to_bev(boxes3d)
    # convert the rotation to a valid range
    rotations = bev_rotated_boxes[:, -1]
    normed_rotations = torch.abs(limit_period(rotations, 0.5, np.pi))

    # find the center of boxes
    conditions = (normed_rotations > np.pi / 4)[..., None]
    bboxes_xywh = torch.where(conditions, bev_rotated_boxes[:, [0, 1, 3, 2]],
                              bev_rotated_boxes[:, :4])

    centers = bboxes_xywh[:, :2]
    dims = bboxes_xywh[:, 2:]
    bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
    return bev_boxes


class Anchor3DRangeGenerator(object):
    """3D Anchor Generator by range.

    This anchor generator generates anchors by the given range in different
    feature levels.

    Args:
        ranges (list[list[float]]): Ranges of different anchors.
            The ranges are the same across different feature levels. But may
            vary for different anchor sizes if size_per_range is True.
        sizes (list[list[float]]): 3D sizes of anchors.
        rotations (list[float]): Rotations of anchors in a feature grid.
    """

    def __init__(self,
                 ranges,
                 sizes=[[1.6, 3.9, 1.56]],
                 rotations=[0, 1.5707963]):

        if len(sizes) != len(ranges):
            assert len(ranges) == 1
            ranges = ranges * len(sizes)
        assert len(ranges) == len(sizes)

        self.sizes = sizes
        self.ranges = ranges
        self.rotations = rotations

    @property
    def num_base_anchors(self):
        """list[int]: Total number of base anchors in a feature grid."""
        num_rot = len(self.rotations)
        num_size = torch.tensor(self.sizes).reshape(-1, 3).size(0)
        return num_rot * num_size

    def grid_anchors(self, featmap_size, device='cuda'):
        """Generate grid anchors of a single level feature map.

        This function is usually called by method ``self.grid_anchors``.

        Args:
            featmap_size (tuple[int]): Size of the feature map.
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature map.
        """
        mr_anchors = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchors.append(
                self.anchors_single_range(featmap_size,
                                          anchor_range,
                                          anchor_size,
                                          self.rotations,
                                          device=device))
        mr_anchors = torch.cat(mr_anchors, dim=-3)
        return mr_anchors

    def anchors_single_range(self,
                             feature_size,
                             anchor_range,
                             sizes=[[1.6, 3.9, 1.56]],
                             rotations=[0, 1.5707963],
                             device='cuda'):
        """Generate anchors in a single range.

        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (torch.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            sizes (list[list] | np.ndarray | torch.Tensor): Anchor size with
                shape [N, 3], in order of x, y, z.
            rotations (list[float] | np.ndarray | torch.Tensor): Rotations of
                anchors in a single feature grid.
            device (str): Devices that the anchors will be put on.

        Returns:
            torch.Tensor: Anchors with shape \
                [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = torch.tensor(anchor_range, device=device)
        z_centers = torch.linspace(anchor_range[2],
                                   anchor_range[5],
                                   feature_size[0],
                                   device=device)
        y_centers = torch.linspace(anchor_range[1],
                                   anchor_range[4],
                                   feature_size[1],
                                   device=device)
        x_centers = torch.linspace(anchor_range[0],
                                   anchor_range[3],
                                   feature_size[2],
                                   device=device)
        sizes = torch.tensor(sizes, device=device).reshape(-1, 3)
        rotations = torch.tensor(rotations, device=device)

        # torch.meshgrid default behavior is 'id', np's default is 'xy'
        rets = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        # torch.meshgrid returns a tuple rather than list
        rets = list(rets)
        for i in range(len(rets)):
            rets[i] = rets[i].unsqueeze(-2).unsqueeze(-1)

        sizes = sizes.reshape([1, 1, 1, 1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = sizes.repeat(tile_size_shape)
        rets.insert(3, sizes)

        ret = torch.cat(rets, dim=-1).permute([2, 1, 0, 3, 4, 5])
        # [1, 200, 176, N, 2, 7] for kitti after permute
        return ret


class BBoxCoder(object):
    """Bbox Coder for 3D boxes.

    Args:
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(self):
        super(BBoxCoder, self).__init__()

    @staticmethod
    def encode(src_boxes, dst_boxes):
        """Get box regression transformation deltas (dx, dy, dz, dw, dh, dl, dr,
        dv*) that can be used to transform the `src_boxes` into the
        `target_boxes`.

        Args:
            src_boxes (torch.Tensor): source boxes, e.g., object proposals.
            dst_boxes (torch.Tensor): target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas.
        """
        xa, ya, za, wa, la, ha, ra = torch.split(src_boxes, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg = torch.split(dst_boxes, 1, dim=-1)
        za = za + ha / 2
        zg = zg + hg / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt], dim=-1)

    @staticmethod
    def decode(anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
        `boxes`.

        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, w, l, h, r, velo*].

        Returns:
            torch.Tensor: Decoded boxes.
        """
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)

        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)


def multiclass_nms(boxes, scores, score_thr):
    """Multi-class nms for 3D boxes.

    Args:
        boxes (torch.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        scores (torch.Tensor): Multi-level boxes with shape
            (N, ). N is the number of boxes.
        score_thr (float): Score threshold to filter boxes with low
            confidence.

    Returns:
        list[torch.Tensor]: Return a list of indices after nms,
            with an entry for each class.
    """
    idxs = []
    for i in range(scores.shape[1]):
        cls_inds = scores[:, i] > score_thr
        if not cls_inds.any():
            idxs.append(
                torch.tensor([], dtype=torch.long, device=cls_inds.device))
            continue

        orig_idx = torch.arange(cls_inds.shape[0],
                                device=cls_inds.device,
                                dtype=torch.long)[cls_inds]
        _scores = scores[cls_inds, i]
        _boxes = boxes[cls_inds, :]
        _bev = xywhr_to_xyxyr(box3d_to_bev(_boxes))

        idx = nms(_bev, _scores, 0.01)

        idxs.append(orig_idx[idx])

    return idxs


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or "iof" (intersection over
            foreground).
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] -
                                                   bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] -
                                                   bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
