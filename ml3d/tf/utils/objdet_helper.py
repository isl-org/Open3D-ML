import numpy as np
import tensorflow as tf

from functools import partial

from open3d.ml.tf.ops import nms


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (tf.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        tf.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = tf.expand_dims(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = tf.reshape(tf.range(max_num, dtype=tf.int64), (max_num_shape))
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = tf.cast(actual_num, tf.int64) > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (tf.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        tf.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    return val - tf.floor(val / period + offset) * period


def xywhr_to_xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (tf.Tensor): Rotated boxes in XYWHR format.

    Returns:
        tf.Tensor: Converted boxes in XYXYR format.
    """
    transform = tf.constant([
        [1.0, 0.0, -0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.5, 0.0],
        [1.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    return tf.linalg.matvec(transform, boxes_xywhr)


def box3d_to_bev(boxes3d):
    """Convert rotated 3d boxes in XYZWHDR format to BEV in XYWHR format.

    Args:
        boxes3d (torch.Tensor): Rotated boxes in XYZWHDR format.

    Returns:
        tf.Tensor: Converted BEV boxes in XYWHR format.
    """
    return tf.gather(boxes3d, [0, 1, 3, 4, 6], axis=-1)


def box3d_to_bev2d(boxes3d):
    """Convert rotated 3d boxes in XYZWHDR format to neareset BEV without
    rotation.

    Args:
        boxes3d (torch.Tensor): Rotated boxes in XYZWHDR format.

    Returns:
        tf.Tensor: Converted BEV boxes in XYWH format.
    """
    # Obtain BEV boxes with rotation in XYWHR format
    bev_rotated_boxes = box3d_to_bev(boxes3d)
    # convert the rotation to a valid range
    rotations = bev_rotated_boxes[:, -1]
    normed_rotations = tf.abs(limit_period(rotations, 0.5, np.pi))

    # find the center of boxes
    conditions = (normed_rotations > np.pi / 4)[..., None]
    bboxes_xywh = tf.where(conditions,
                           tf.gather(bev_rotated_boxes, [0, 1, 3, 2], axis=-1),
                           bev_rotated_boxes[:, :4])

    centers = bboxes_xywh[:, :2]
    dims = bboxes_xywh[:, 2:]
    bev_boxes = tf.concat([centers - dims / 2, centers + dims / 2], axis=-1)
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
        num_size = tf.reshape(tf.constant(self.sizes), (-1, 3)).shape[0]
        return num_rot * num_size

    def grid_anchors(self, featmap_size):
        """Generate grid anchors of a single level feature map.

        This function is usually called by method ``self.grid_anchors``.

        Args:
            featmap_size (tuple[int]): Size of the feature map.

        Returns:
            tf.Tensor: Anchors in the overall feature map.
        """
        mr_anchors = []
        for anchor_range, anchor_size in zip(self.ranges, self.sizes):
            mr_anchors.append(
                self.anchors_single_range(featmap_size, anchor_range,
                                          anchor_size, self.rotations))
        mr_anchors = tf.concat(mr_anchors, axis=-3)
        return mr_anchors

    def anchors_single_range(self,
                             feature_size,
                             anchor_range,
                             sizes=[[1.6, 3.9, 1.56]],
                             rotations=[0, 1.5707963]):
        """Generate anchors in a single range.

        Args:
            feature_size (list[float] | tuple[float]): Feature map size. It is
                either a list of a tuple of [D, H, W](in order of z, y, and x).
            anchor_range (tf.Tensor | list[float]): Range of anchors with
                shape [6]. The order is consistent with that of anchors, i.e.,
                (x_min, y_min, z_min, x_max, y_max, z_max).
            sizes (list[list] | np.ndarray | tf.Tensor): Anchor size with
                shape [N, 3], in order of x, y, z.
            rotations (list[float] | np.ndarray | tf.Tensor): Rotations of
                anchors in a single feature grid.

        Returns:
            tf.Tensor: Anchors with shape \
                [*feature_size, num_sizes, num_rots, 7].
        """
        if len(feature_size) == 2:
            feature_size = [1, feature_size[0], feature_size[1]]
        anchor_range = tf.constant(anchor_range)
        z_centers = tf.linspace(anchor_range[2], anchor_range[5],
                                feature_size[0])
        y_centers = tf.linspace(anchor_range[1], anchor_range[4],
                                feature_size[1])
        x_centers = tf.linspace(anchor_range[0], anchor_range[3],
                                feature_size[2])
        sizes = tf.constant(sizes)
        rotations = tf.constant(rotations)

        # torch.meshgrid default behavior is 'id', tf's default is 'xy'
        rets = tf.meshgrid(x_centers,
                           y_centers,
                           z_centers,
                           rotations,
                           indexing='ij')
        for i in range(len(rets)):
            rets[i] = tf.expand_dims(tf.expand_dims(rets[i], -2), -1)

        tile_size_shape = list(rets[0].shape)
        tile_size_shape[-1] = sizes.shape[-1]
        sizes = tf.zeros(tile_size_shape) + sizes
        rets.insert(3, sizes)

        ret = tf.transpose(tf.concat(rets, axis=-1), perm=(2, 1, 0, 3, 4, 5))
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
            src_boxes (tf.Tensor): source boxes, e.g., object proposals.
            dst_boxes (tf.Tensor): target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            tf.Tensor: Box transformation deltas.
        """
        xa, ya, za, wa, la, ha, ra = tf.split(src_boxes, 7, axis=-1)
        xg, yg, zg, wg, lg, hg, rg = tf.split(dst_boxes, 7, axis=-1)
        za = za + ha / 2
        zg = zg + hg / 2
        diagonal = tf.sqrt(la**2 + wa**2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = tf.math.log(lg / la)
        wt = tf.math.log(wg / wa)
        ht = tf.math.log(hg / ha)
        rt = rg - ra
        return tf.concat([xt, yt, zt, wt, lt, ht, rt], axis=-1)

    @staticmethod
    def decode(anchors, deltas):
        """Apply transformation `deltas` (dx, dy, dz, dw, dh, dl, dr, dv*) to
        `boxes`.

        Args:
            anchors (tf.Tensor): Parameters of anchors with shape (N, 7).
            deltas (tf.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, w, l, h, r, velo*].

        Returns:
            tf.Tensor: Decoded boxes.
        """
        xa, ya, za, wa, la, ha, ra = tf.split(anchors, 7, axis=-1)
        xt, yt, zt, wt, lt, ht, rt = tf.split(deltas, 7, axis=-1)

        za = za + ha / 2
        diagonal = tf.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = tf.exp(lt) * la
        wg = tf.exp(wt) * wa
        hg = tf.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        return tf.concat([xg, yg, zg, wg, lg, hg, rg], axis=-1)


def multiclass_nms(boxes, scores, score_thr):
    """Multi-class nms for 3D boxes.

    Args:
        boxes (tf.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        scores (tf.Tensor): Multi-level boxes with shape
            (N, ). N is the number of boxes.
        score_thr (float): Score threshold to filter boxes with low
            confidence.

    Returns:
        list[tf.Tensor]: Return a list of indices after nms,
            with an entry for each class.
    """
    idxs = []
    for i in range(scores.shape[1]):
        cls_inds = tf.where(scores[:, i] > score_thr)[:, 0]

        _scores = tf.gather(scores, cls_inds)[:, i]
        _boxes = tf.gather(boxes, cls_inds)
        _bev = xywhr_to_xyxyr(tf.gather(_boxes, [0, 1, 3, 4, 6], axis=1))

        idx = nms(_bev, _scores, 0.01)
        idxs.append(tf.gather(cls_inds, idx))

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
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimension is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[-1] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[-1] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return tf.zeros(batch_shape + (rows,))
        else:
            return tf.zeros(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] -
                                                   bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] -
                                                   bboxes2[..., 1])

    if is_aligned:
        lt = tf.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = tf.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = tf.nn.relu(rb - lt)  # [B, rows, 2] clip to zero
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = tf.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = tf.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = tf.maximum(bboxes1[..., :, None, :2],
                        bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = tf.minimum(bboxes1[..., :, None, 2:],
                        bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = tf.nn.relu(rb - lt)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = tf.minimum(bboxes1[..., :, None, :2],
                                     bboxes2[..., None, :, :2])
            enclosed_rb = tf.maximum(bboxes1[..., :, None, 2:],
                                     bboxes2[..., None, :, 2:])

    eps = tf.constant([eps])
    union = tf.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = tf.nn.relu(enclosed_rb - enclosed_lt)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = tf.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
