import numpy as np
import random
from .operations import create_3D_rotations


def trans_normalize(pc, feat, t_normalize):
    if t_normalize is None or t_normalize.get('method', None) is None:
        return pc, feat

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


def trans_augment(points, t_augment):
    """Implementation of an augmentation transform for point clouds."""

    if t_augment is None or not t_augment.get('turn_on', True):
        return points

    # Initialize rotation matrix
    R = np.eye(points.shape[1])

    if points.shape[1] == 3:
        rotation_method = t_augment.get('rotation_method', None)
        if rotation_method == 'vertical':

            # Create random rotations
            theta = np.random.rand() * 2 * np.pi
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        elif rotation_method == 'all':

            # Choose two random angles for the first vector in polar coordinates
            theta = np.random.rand() * 2 * np.pi
            phi = (np.random.rand() - 0.5) * np.pi

            # Create the first vector in carthesian coordinates
            u = np.array([
                np.cos(theta) * np.cos(phi),
                np.sin(theta) * np.cos(phi),
                np.sin(phi)
            ])

            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)),
                                    np.reshape(alpha, (1, -1)))[0]

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
