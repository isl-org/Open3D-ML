import logging as log
import numpy as np
import random
from .operations import *


def trans_normalize(pc, feat, t_normalize):
    dim = t_normalize.get('recentering', [0, 1, 2])
    pc[:, dim] = pc[:, dim] - pc.mean(0)[dim]

    if t_normalize.get('method', None):
        method = t_normalize['method']
        if method == 'linear':
            if t_normalize.get('normalize_points', False):
                pc -= pc.mean()
                pc /= (pc.max(0) - pc.min(0)).max()

            if feat is not None:
                feat_bias = t_normalize.get('feat_bias', 0)
                feat_scale = t_normalize.get('feat_scale', 1)
                feat -= feat_bias
                feat /= feat_scale
        elif method == 'coords_only':
            feat = None

    return pc, feat


def create_random_rotation(rotation_method='all',
                           rotation_range=[np.pi, np.pi / 2, np.pi]):
    """ Create rotation matrix for a random rotation in a bounded interval

    Args:
        rotation_method (str): 'vertical' or 'z' for rotation around Z axis only.
            'all' for 3D rotation
        rotation_range (ArrayLike[3]): Rotation is bounded in the interval
            [-theta, theta] around each axis.

    Returns:
        ArrayLike[3,3]: Rotation matrix
    """

    if rotation_method in ('vertical', 'z'):

        # Create random rotations
        theta = (2 * np.random.rand() - 1) * rotation_range[2]
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    elif rotation_method == 'all':

        rnd_num = np.full((3,), np.pi / 2)  # (2 * np.random.rand(3) - 1)
        # Choose two random angles for the first vector in polar coordinates
        theta = rnd_num[0] * rotation_range[0]
        phi = rnd_num[1] * rotation_range[1]

        # Create the first vector in Cartesian coordinates
        u = np.array([
            np.cos(theta) * np.cos(phi),
            np.sin(theta) * np.cos(phi),
            np.sin(phi)
        ])

        # Choose a random rotation angle
        alpha = rnd_num[2] * rotation_range[2]

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(np.reshape(u, (1, -1)),
                                np.reshape(alpha, (1, -1)))[0]
    else:
        raise ValueError(f"Unknown rotation method {rotation_method}.")

    return R


def trans_augment(points, t_augment):
    """Implementation of an augmentation transform for point clouds."""

    if t_augment is None or not t_augment.get('turn_on', True):
        return points

    if points.shape[1] == 3:
        rotation_method = t_augment.get('rotation_method', None)
        R = create_random_rotation(rotation_method)

    R = R.astype(np.float32)

    # Choose random scales for each example
    scale_anisotropic = t_augment.get('scale_anisotropic', False)
    min_s = t_augment.get('min_s', 1.)
    max_s = t_augment.get('max_s', 1.)
    if scale_anisotropic:
        scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
    else:
        scale = np.random.rand() * (max_s - min_s) - min_s

    # TODO: add symmetric augmentation
    # # Add random symmetries to the scale factor
    # symmetries = []
    # sym = t_augment.get('symmetries', [False, False, False])
    # for i in range(3):
    #     if sym[i]:
    #         symmetries.append(tf.round(tf.random_uniform((1, 1))) * 2 - 1)
    #     else:
    #         symmetries.append(tf.ones([1, 1], dtype=tf.float32))

    # symmetries = np.array(symmetries).astype(np.int32)
    # symmetries = symmetries * np.random.randint(2, size=points.shape[1])
    # scale = (scale * (1 - symmetries * 2)).astype(np.float32)

    noise_level = t_augment.get('noise_level', 0.001)
    noise = (np.random.randn(points.shape[0], points.shape[1]) *
             noise_level).astype(np.float32)

    augmented_points = np.sum(np.expand_dims(points, 2) * R,
                              axis=1) * scale + noise

    return augmented_points.astype(np.float32)


def trans_crop_pc(points, feat, labels, search_tree, pick_idx, num_points):
    # crop a fixed size point cloud for training
    center_point = points[pick_idx, :].reshape(1, -1)

    if (points.shape[0] < num_points):
        select_idx = np.array(range(points.shape[0]))
        diff = num_points - points.shape[0]
        select_idx = list(select_idx) + list(random.choices(select_idx, k=diff))
        random.shuffle(select_idx)
    else:
        select_idx = search_tree.query(center_point, k=num_points)[1][0]

    random.shuffle(select_idx)
    select_points = points[select_idx]
    select_labels = labels[select_idx]
    if (feat is None):
        select_feat = None
    else:
        select_feat = feat[select_idx]

    select_points = select_points - center_point

    return select_points, select_feat, select_labels, select_idx


def in_range_bev(box_range, box):
    return (box[0] > box_range[0]) & (box[1] > box_range[1]) & (
        box[0] < box_range[2]) & (box[1] < box_range[3])


class ObjdetAugmentation():
    """Class consisting different augmentation for Object Detection"""

    @staticmethod
    def PointShuffle(data):
        np.random.shuffle(data['point'])

        return data

    @staticmethod
    def ObjectRangeFilter(data, pcd_range):
        """ used last to ensure model gets points and boxes only inside the
        declared range after other augmentations."""
        pcd_range = np.array(pcd_range)
        bev_range = pcd_range[[0, 1, 3, 4]]

        filtered_boxes = []
        for box in data['bbox_objs']:
            if in_range_bev(bev_range, box.to_xyzwhlr()):
                filtered_boxes.append(box)
        data['bbox_objs'] = filtered_boxes

        pcd = data['point']
        in_range_idx = np.logical_and.reduce((
            pcd[:, 0] > pcd_range[0],
            pcd[:, 0] < pcd_range[3],
            pcd[:, 1] > pcd_range[1],
            pcd[:, 1] < pcd_range[4],
            pcd[:, 2] > pcd_range[2],
            pcd[:, 2] < pcd_range[5],
        ))
        data['point'] = pcd[in_range_idx, :]
        return data

    @staticmethod
    def ObjectSample(data, db_boxes_dict, sample_dict):
        rate = 1.0
        points = data['point']
        bboxes = data['bbox_objs']

        gt_labels_3d = [box.label_class for box in data['bbox_objs']]

        sampled_num_dict = {}

        for class_name in sample_dict.keys():
            max_sample_num = sample_dict[class_name]

            existing = np.sum([n == class_name for n in gt_labels_3d])
            sampled_num = int(max_sample_num - existing)
            sampled_num = np.round(rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num

        sampled = []
        for class_name in sampled_num_dict.keys():
            sampled_num = sampled_num_dict[class_name]
            if sampled_num < 0:
                continue

            sampled_cls = sample_class(class_name, sampled_num, bboxes,
                                       db_boxes_dict[class_name])
            sampled += sampled_cls
            bboxes = bboxes + sampled_cls

        if len(sampled) != 0:
            sampled_points = np.concatenate(
                [box.points_inside_box for box in sampled], axis=0)
            points = remove_points_in_boxes(points, sampled)
            points = np.concatenate([sampled_points, points], axis=0)

        return {'point': points, 'bbox_objs': bboxes, 'calib': data['calib']}

    @staticmethod
    def ObjectNoise(input,
                    trans_std=[0.25, 0.25, 0.25],
                    rot_range=[-0.15707963267, 0.15707963267],
                    num_try=100):
        raise NotImplementedError

    @staticmethod
    def SensorNoise(data,
                    type='stereo',
                    quad_mult_coeff=0.003,
                    linear_coeff=0.0017):
        pcd = data['point']
        if type == 'stereo':
            pcd[:, 2] += np.random.rand(
                pcd.shape[0]) * pcd[:, 2]**2 * quad_mult_coeff
        elif type == 'lidar':
            pcd[:, 2] *= 1 + np.random.rand(pcd.shape[0]) * linear_coeff

        data.update({'point': pcd})
        return data

    @staticmethod
    def CameraDolly(data,
                    trans_range=(-1, 1),
                    motion_axis='Z',
                    point_cloud_range=[-2, -1.5, 0, 2, 1.5, 3.5]):
        axis_idx = "XYZ".find(motion_axis)
        if axis_idx < 0:
            log.error(f"Unsupported motion_axis {motion_axis}")
        if trans_range[1] > point_cloud_range[3 + axis_idx]:
            log.error(
                "Max dolly translation must be within the point_cloud_range")
        pcd = data['point']
        trans_range[0] = max(trans_range[0], 0.5 - np.min(pcd[:, axis_idx]))
        shift = trans_range[0] + np.random.rand() * (trans_range[1] -
                                                     trans_range[0])
        for k in range(len(data['bbox_objs'])):
            data['bbox_objs'][k].center[axis_idx] += shift
        pcd[:, axis_idx] += shift
        sample_ratio = (point_cloud_range[3 + axis_idx] +
                        shift) / point_cloud_range[3 + axis_idx]
        if shift > 0:  # dolly out -> subsample points
            pcd = pcd[np.arange(0, pcd.shape[0] -
                                1e-3, sample_ratio).astype(int), :]
        elif shift < 0:  # dolly in -> crop and nearest neighbor upsample
            new_min = np.array(point_cloud_range[:3]) * sample_ratio
            new_max = np.array(point_cloud_range[3:6]) * sample_ratio
            dim0, dim1 = {0, 1, 2} - {axis_idx}
            in_view_idx = np.logical_and.reduce(
                (pcd[:, dim0] > new_min[dim0], pcd[:, dim0] < new_max[dim0],
                 pcd[:, dim1] > new_min[dim1], pcd[:, dim1] < new_max[dim1]))
            pcd = pcd[in_view_idx, :]
            # TODO(Sameer): Linear interpolation for PCD upsampling
            pcd = pcd[np.arange(0, pcd.shape[0] -
                                1e-3, sample_ratio).astype(int), :]

        data.update({'point': pcd})
        return data

    @staticmethod
    def Rotation(data, rotation_method='all', rotation_range=[180, 90, 180]):
        """ Output points may be out of range """

        R = create_random_rotation(rotation_method, np.deg2rad(rotation_range))
        pcd = data['point']
        assert pcd.shape[1] == 3, "data['point'] shape is not (N,3)"
        data['point'] = pcd @ R
        R_homo = np.eye(4)
        R_homo[:3, :3] = R
        for k in range(len(data['bbox_objs'])):
            data['bbox_objs'][k].transform(R_homo)
            data['bbox_objs'][k].get_yaw()

        return data
