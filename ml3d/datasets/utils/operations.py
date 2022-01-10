import numpy as np
import random
import copy
import math
from scipy.spatial import ConvexHull

from open3d.ml.contrib import iou_bev_cpu as iou_bev


def create_3D_rotations(axis, angle):
    """Create rotation matrices from a list of axes and angles. Code from
    wikipedia on quaternions.

    Args:
        axis: float32[N, 3]
        angle: float32[N,]

    Returns:
        float32[N, 3, 3]
    """
    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([
        t1 + t2 * t3, t7 - t9, t11 + t12, t7 + t9, t1 + t2 * t15, t19 - t20,
        t11 - t12, t19 + t20, t1 + t2 * t24
    ],
                 axis=1)

    return np.reshape(R, (-1, 3, 3))


def projection_matrix_to_CRT_kitti(proj):
    """Split projection matrix of kitti.

    P = C @ [R|T]
    C is upper triangular matrix, so we need to inverse CR and use QR
    stable for all kitti camera projection matrix.

    Args:
        proj (p.array, shape=[4, 4]): Intrinsics of camera.

    Returns:
        tuple[np.ndarray]: Splited matrix of C, R and T.
    """
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    """Get frustum corners in camera coordinates.

    Args:
        bbox_image (list[int]): box in image coordinates.
        C (np.ndarray): Intrinsics.
        near_clip (float): Nearest distance of frustum.
        far_clip (float): Farthest distance of frustum.

    Returns:
        np.ndarray, shape=[8, 3]: coordinates of frustum corners.
    """
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4,
                        dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]], dtype=C.dtype)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
    ret_xy = np.concatenate([near_box_corners, far_box_corners],
                            axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


def camera_to_lidar(points, world_cam):
    """Convert points in camera coordinate to lidar coordinate.

    Args:
        points (np.ndarray, shape=[N, 3]): Points in camera coordinate.
        world_cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinates to lidar coordinates.

    Returns:
        np.ndarray, shape=[N, 3]: Points in lidar coordinates.
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv(world_cam)
    return lidar_points[..., :3]


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2**ndim), [2] * ndim),
                            axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners


def rotation_3d_in_axis(points, angles, axis=2):
    """Rotate points in specific axis.

    Args:
        points (np.ndarray, shape=[N, point_size, 3]]):
        angles (np.ndarray, shape=[N]]):
        axis (int): Axis to rotate at.

    Returns:
        np.ndarray: Rotated points.
    """
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError('axis should in range')

    return np.einsum('aij,jka->aik', points, rot_mat_T)


def rotation_2d(points, angles):
    """Rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (np.ndarray): Points to be rotated with shape \
            (N, point_size, 2).
        angles (np.ndarray): Rotation angle with shape (N).

    Returns:
        np.ndarray: Same shape as points.
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 1.0, 0.5)):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray): Rotation_y in kitti label file with shape (N).
        origin (list or array or float): Origin point relate to smallest point.
            use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.

    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles)
    corners += centers.reshape([-1, 1, 3])
    return corners


def center_to_corner_box2d(boxes, origin=0.5):
    """Convert kitti locations, dimensions and angles to corners.

    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 2).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 2).
        angles (np.ndarray): Rotation_y in kitti label file with shape (N).

    Returns:
        np.ndarray: Corners with the shape of (N, 4, 2).
    """
    if len(boxes) == 0:
        return np.zeros((0, 4, 2))
    flat_boxes = np.array([box.to_xyzwhlr() for box in boxes])
    centers = flat_boxes[:, 0:2]
    dims = flat_boxes[:, 3:5]
    angles = flat_boxes[:, 6]
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


def corner_to_surfaces_3d(corners):
    """Convert 3d box corners from corner function above to surfaces that normal
    vectors all direct to internal.

    Args:
        corners (np.ndarray): 3D box corners with shape of (N, 8, 3).

    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces


def surface_equ_3d(polygon_surfaces):
    """Compute normal vectors for polygon surfaces.

    Args:
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.

    Returns:
        tuple: normal vector and its direction.
    """
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - \
        polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


def points_in_convex_polygon_3d(points, polygon_surfaces, num_surfaces=None):
    """Check points is in 3d convex polygons.

    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of \
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3). \
            All surfaces' normal vector must direct to internal. \
            Max_num_points_of_surface must at least 3.
        num_surfaces (np.ndarray): Number of surfaces a polygon contains \
            shape of (num_polygon).

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    # num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]

    # expand dims for broadcasting
    points = np.reshape(points, (num_points, 1, 1, 3))
    normal_vec = np.reshape(normal_vec, (1, num_polygons, max_num_surfaces, 3))
    num_surfaces = np.reshape(num_surfaces, (num_polygons, 1))

    sign = np.sum(points * normal_vec, axis=-1) + d

    out_range = np.arange(max_num_surfaces) >= num_surfaces
    out_range = np.reshape(out_range, (1, num_polygons, max_num_surfaces))
    ret = np.all(sign < 0 | out_range, axis=-1)
    return ret


def points_in_box(points,
                  rbbox,
                  origin=(0.5, 0.5, 0),
                  camera_frame=False,
                  cam_world=None):
    """Check points in rotated bbox and return indices.

    If `rbbox` is in camera frame, it is first converted to world frame using
    `cam_world`. Returns a 2D array classifying each point for each box.

    Args:
        points (np.ndarray, shape=[N, 3+dim]): Points to query.
        rbbox (np.ndarray, shape=[M, 7]): Boxes3d with rotation (camera/world frame).
        origin (tuple[int]): Indicate the position of box center.
        camera_frame: True if `rbbox` are in camera frame(like kitti format, where y
          coordinate is height), False for [x, y, z, dx, dy, dz, yaw] format.
        cam_world: camera to world transformation matrix. Required when `camera_frame` is True.

    Returns:
        np.ndarray, shape=[N, M]: Indices of points in each box.
    """
    if len(rbbox) == 0:
        return np.zeros((0, 7))
    if camera_frame:
        assert cam_world is not None, "Provide cam_to_world matrix if points are in camera frame."

        # transform in world space
        points = np.hstack(
            (points, np.ones((points.shape[0], 1), dtype=np.float32)))
        points = np.matmul(points, cam_world)[..., :3]

    rbbox = np.array(rbbox)
    rbbox_corners = center_to_corner_box3d(rbbox[:, :3],
                                           rbbox[:, 3:6],
                                           rbbox[:, 6],
                                           origin=origin)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d(points[:, :3], surfaces)
    return indices


def filter_by_min_points(bboxes, min_points_dict):
    """Filter ground truths by number of points in the bbox."""
    filtered_boxes = []

    for box in bboxes:
        if box.label_class in min_points_dict.keys():
            if box.points_inside_box.shape[0] > min_points_dict[
                    box.label_class]:
                filtered_boxes.append(box)
        else:
            filtered_boxes.append(box)

    return filtered_boxes


def random_sample(files, num):
    if len(files) <= num:
        return files

    return random.sample(files, num)


def corner_to_standup_nd_jit(boxes_corner):
    """Convert boxes_corner to aligned (min-max) boxes.

    Args:
        boxes_corner (np.ndarray, shape=[N, 2**dim, dim]): Boxes corners.

    Returns:
        np.ndarray, shape=[N, dim*2]: Aligned (min-max) boxes.
    """
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


def box_collision_test(boxes, qboxes):
    """Box collision test.

    Args:
        boxes (np.ndarray): Corners of current boxes.
        qboxes (np.ndarray): Boxes to be avoid colliding.
    """
    boxes = np.array([box.to_xyzwhlr() for box in boxes], dtype=np.float32)
    qboxes = np.array([box.to_xyzwhlr() for box in qboxes], dtype=np.float32)

    boxes = boxes[:, [0, 1, 3, 4, 6]]
    qboxes = qboxes[:, [0, 1, 3, 4, 6]]

    coll_mat = iou_bev(boxes, qboxes)

    coll_mat[coll_mat != 0] = 1

    return coll_mat.astype(np.bool)


def sample_class(class_name, num, gt_boxes, db_boxes):
    if num == 0:
        return []

    sampled = random_sample(db_boxes, num)
    sampled = copy.deepcopy(sampled)

    num_gt = len(gt_boxes)
    num_sampled = len(sampled)

    gt_boxes_bev = center_to_corner_box2d(gt_boxes)

    boxes = (gt_boxes + sampled).copy()

    coll_mat = box_collision_test(boxes, boxes)

    diag = np.arange(len(boxes))
    coll_mat[diag, diag] = False

    valid_samples = []
    for i in range(num_gt, num_gt + num_sampled):
        if coll_mat[i].any():
            coll_mat[i] = False
            coll_mat[:, i] = False
        else:
            valid_samples.append(sampled[i - num_gt])

    return valid_samples


def remove_points_in_boxes(points, boxes):
    """Remove the points in the sampled bounding boxes.

    Args:
        points (np.ndarray): Input point cloud array.
        boxes (np.ndarray): Sampled ground truth boxes.

    Returns:
        np.ndarray: Points with those in the boxes removed.
    """
    flat_boxes = [box.to_xyzwhlr() for box in boxes]
    masks = points_in_box(points, flat_boxes)
    points = points[np.logical_not(masks.any(-1))]

    return points


def get_min_bbox(points):
    """Return minimum bounding box encapsulating points.

    Args:
        points (np.ndarray): Input point cloud array.

    Returns:
        np.ndarray: 3D BEV bounding box (x, y, z, w, h, l, yaw).
    """
    points = points.copy()
    h_min = np.min(points[:, 2])
    h_max = np.max(points[:, 2])

    points = points[:, :2]

    # cov_hull = ConvexHull(points)
    # points = cov_hull.points[cov_hull.vertices]

    cov_points = np.cov(points, rowvar=False, bias=True)
    val, vect = np.linalg.eig(cov_points)
    tvect = np.transpose(vect)

    points_rot = np.dot(points, np.linalg.inv(tvect))

    min_a = np.min(points_rot, axis=0)
    max_a = np.max(points_rot, axis=0)

    diff = max_a - min_a

    center = min_a + diff * 0.5
    center = np.dot(center, tvect)

    center = np.array([center[0], center[1], (h_min + h_max) * 0.5])

    width = diff[0]
    length = diff[1]
    height = h_max - h_min

    yaw = math.atan(tvect[0, 1] / tvect[0, 0])

    return [center[0], center[1], center[2], width, height, length, yaw]
