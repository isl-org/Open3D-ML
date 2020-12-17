import numpy as np
import os, argparse, pickle, sys
import open3d.core as o3c

from os.path import exists, join, isfile, dirname, abspath, split
from open3d.ml.contrib import subsample
from open3d.ml.contrib import knn_search


class DataProcessing:

    @staticmethod
    def grid_subsampling(points,
                         features=None,
                         labels=None,
                         grid_size=0.1,
                         verbose=0):
        """
        CPP wrapper for a grid subsampling (method = barycenter for points and features)
        :param points: (N, 3) matrix of input points
        :param features: optional (N, d) matrix of features (floating number)
        :param labels: optional (N,) matrix of integer labels
        :param sampleDl: parameter defining the size of grid voxels
        :param verbose: 1 to display
        :return: subsampled points, with features and/or labels depending of the input
        """

        if (features is None) and (labels is None):
            return subsample(points, sampleDl=grid_size, verbose=verbose)
        elif (labels is None):
            return subsample(points,
                             features=features,
                             sampleDl=grid_size,
                             verbose=verbose)
        elif (features is None):
            return subsample(points,
                             classes=labels,
                             sampleDl=grid_size,
                             verbose=verbose)
        else:
            return subsample(points,
                             features=features,
                             classes=labels,
                             sampleDl=grid_size,
                             verbose=verbose)

    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename,
                            header=None,
                            delim_whitespace=True,
                            dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename,
                               header=None,
                               delim_whitespace=True,
                               dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        # points = scan[:, 0:3]  # get xyz
        points = scan
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """

        neighbor_idx = knn_search(o3c.Tensor.from_numpy(query_pts),
                                  o3c.Tensor.from_numpy(support_pts),
                                  k).numpy()

        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def IoU_from_confusions(confusions):
        """
        Computes IoU from confusion matrices.
        :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes
        :return: ([..., n_c] np.float32) IoU score
        """

        # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
        # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        # Compute IoU
        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        # Compute mIoU with only the actual classes
        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
        IoU += mask * mIoU
        return IoU

    @staticmethod
    def Acc_from_confusions(confusions):
        return confusions.diagonal() / confusions.sum(axis=0)

    @staticmethod
    def get_class_weights(num_per_class):
        # pre-calculate the number of points in each category
        num_per_class = np.array(num_per_class, dtype=np.float32)

        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)

        return np.expand_dims(ce_label_weight, axis=0)

    @staticmethod
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

    @staticmethod
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
            [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]],
            dtype=C.dtype)
        near_box_corners = (box_corners - u0v0) / np.array(
            [fku / near_clip, -fkv / near_clip], dtype=C.dtype)
        far_box_corners = (box_corners - u0v0) / np.array(
            [fku / far_clip, -fkv / far_clip], dtype=C.dtype)
        ret_xy = np.concatenate([near_box_corners, far_box_corners],
                                axis=0)  # [8, 2]
        ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
        return ret_xyz

    @staticmethod
    def camera_to_lidar(points, r_rect, velo2cam):
        """Convert points in camera coordinate to lidar coordinate.
        Args:
            points (np.ndarray, shape=[N, 3]): Points in camera coordinate.
            r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
                specific camera coordinate (e.g. CAM2) to CAM0.
            velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
                camera coordinate to lidar coordinate.
        Returns:
            np.ndarray, shape=[N, 3]: Points in lidar coordinate.
        """
        points_shape = list(points.shape[0:-1])
        if points.shape[-1] == 3:
            points = np.concatenate(
                [points, np.ones(points_shape + [1])], axis=-1)
        lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
        return lidar_points[..., :3]

    @staticmethod
    def corner_to_surfaces_3d(corners):
        """Convert 3d box corners from corner function above to surfaces that
        normal vectors all direct to internal.
        Args:
            corners (np.ndarray): 3d box corners with the shape of (N, 8, 3).
        Returns:
            np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
        """
        # box_corners: [N, 8, 3], must from corner functions in this module
        num_boxes = corners.shape[0]
        surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
        corner_idxes = np.array([
            0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6,
            7
        ]).reshape(6, 4)
        for i in range(num_boxes):
            for j in range(6):
                for k in range(4):
                    surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
        return surfaces

    @staticmethod
    def surface_equ_3d(polygon_surfaces):
        """
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

    @staticmethod
    def points_in_convex_polygon_3d(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
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
        max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[
            1:3]
        # num_points = points.shape[0]
        num_polygons = polygon_surfaces.shape[0]
        if num_surfaces is None:
            num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
        normal_vec, d = DataProcessing.surface_equ_3d(
            polygon_surfaces[:, :, :3, :])
        # normal_vec: [num_polygon, max_num_surfaces, 3]
        # d: [num_polygon, max_num_surfaces]
        max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[
            1:3]
        num_points = points.shape[0]
        num_polygons = polygon_surfaces.shape[0]
        ret = np.ones((num_points, num_polygons), dtype=np.bool_)
        sign = 0.0
        for i in range(num_points):
            for j in range(num_polygons):
                for k in range(max_num_surfaces):
                    if k > num_surfaces[j]:
                        break
                    sign = (points[i, 0] * normal_vec[j, k, 0] +
                            points[i, 1] * normal_vec[j, k, 1] +
                            points[i, 2] * normal_vec[j, k, 2] + d[j, k])
                    if sign >= 0:
                        ret[i, j] = False
                        break
        return ret

    @staticmethod
    def remove_outside_points(points, rect, Trv2c, P2, image_shape):
        """Remove points which are outside of image.
        Args:
            points (np.ndarray, shape=[N, 3+dims]): Total points.
            rect (np.ndarray, shape=[4, 4]): Matrix to project points in
                specific camera coordinate (e.g. CAM2) to CAM0.
            Trv2c (np.ndarray, shape=[4, 4]): Matrix to project points in
                camera coordinate to lidar coordinate.
            P2 (p.array, shape=[4, 4]): Intrinsics of Camera2.
            image_shape (list[int]): Shape of image.
        Returns:
            np.ndarray, shape=[N, 3+dims]: Filtered points.
        """
        C, R, T = DataProcessing.projection_matrix_to_CRT_kitti(P2)
        image_bbox = [0, 0, image_shape[1], image_shape[0]]
        frustum = DataProcessing.get_frustum(image_bbox, C)
        frustum -= T
        frustum = np.linalg.inv(R) @ frustum.T
        frustum = DataProcessing.camera_to_lidar(frustum.T, rect, Trv2c)
        frustum_surfaces = DataProcessing.corner_to_surfaces_3d(
            frustum[np.newaxis, ...])
        indices = DataProcessing.points_in_convex_polygon_3d(
            points[:, :3], frustum_surfaces)
        points = points[indices.reshape([-1])]
        return points
