import numpy as np
import os, argparse, pickle, sys
import open3d.core as o3c

from os.path import exists, join, isfile, dirname, abspath, split
from open3d.ml.contrib import subsample

from .operations import *


class DataProcessing:

    @staticmethod
    def grid_subsampling(points,
                         features=None,
                         labels=None,
                         grid_size=0.1,
                         verbose=0):
        """CPP wrapper for a grid subsampling (method = barycenter for points and
        features).

        Args:
            points: (N, 3) matrix of input points
            features: optional (N, d) matrix of features (floating number)
            labels: optional (N,) matrix of integer labels
            grid_size: parameter defining the size of grid voxels
            verbose: 1 to display

        Returns:
            Subsampled points, with features and/or labels depending of the input
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
        """KNN search.

        Args:
            support_pts: points you have, N1*3
            query_pts: points you want to know the neighbour index, N2*3
            k: Number of neighbours in knn search

        Returns:
            neighbor_idx: neighboring points indexes, N2*k
        """
        nns = o3c.nns.NearestNeighborSearch(o3c.Tensor.from_numpy(support_pts))
        nns.knn_index()
        idx, dist = nns.knn_search(o3c.Tensor.from_numpy(query_pts), k)

        return idx.numpy().astype(np.int32)

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
        """Computes IoU from confusion matrices.

        Args:
            confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
        the last axes. n_c = number of classes

        Returns:
            ([..., n_c] np.float32) IoU score
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

        return ce_label_weight

    @staticmethod
    def invT(T):
        R = T[:3, :3]  # 3x3 rotation + scale
        t = T[3:, :3]  # 1x3 translation

        R = np.linalg.inv(R)
        t = t @ -R

        M = np.concatenate([R, t], axis=0)
        return np.concatenate([M, [[0], [0], [0], [1]]], axis=1)

    @staticmethod
    def world2cam(points, world_cam):
        # transform in cam space
        points = np.hstack(
            (points, np.ones((points.shape[0], 1), dtype=np.float32)))

        for i in range(len(points) // 10000 + 1):
            points[i * 10000:(i + 1) * 10000] = np.matmul(
                points[i * 10000:(i + 1) * 10000], world_cam)

        return points[..., :3]

    @staticmethod
    def cam2img(points, cam_img):
        # transform in image space
        points = np.hstack(
            (points, np.ones((points.shape[0], 1), dtype=np.float32)))
        for i in range(len(points) // 10000 + 1):
            points[i * 10000:(i + 1) * 10000] = np.matmul(
                points[i * 10000:(i + 1) * 10000], cam_img)
        pts_img = (points[:, :2].T / points[:, 3]).T  # (N, 2)
        depth = points[:, 2] - cam_img[3, 2]  # depth in rect camera coord
        return pts_img, depth

    @staticmethod
    def cam2world(points, world_cam):
        cam_world = DataProcessing.invT(world_cam)

        # transform in world space
        points = np.hstack(
            (points, np.ones((points.shape[0], 1), dtype=np.float32)))
        return np.matmul(points, cam_world)[..., :3]

    @staticmethod
    def remove_outside_points(points, world_cam, cam_img, image_shape):
        """Remove points which are outside of image.

        Args:
            points (np.ndarray, shape=[N, 3+dims]): Total points.
            world_cam (np.ndarray, shape=[4, 4]): Matrix to project points in
                lidar coordinates to camera coordinates.
            cam_img (p.array, shape=[4, 4]): Matrix to project points in
                camera coordinates to image coordinates.
            image_shape (list[int]): Shape of image.

        Returns:
            np.ndarray, shape=[N, 3+dims]: Filtered points.
        """
        pts_cam = DataProcessing.world2cam(points[:, :3], world_cam)
        pts_img, depth = DataProcessing.cam2img(pts_cam, cam_img)

        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0,
                                    pts_img[:, 0] < image_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0,
                                    pts_img[:, 1] < image_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        valid = np.logical_and(val_flag_merge, depth >= 0)

        return points[valid]
