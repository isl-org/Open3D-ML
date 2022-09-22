try:
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import range_image_utils, transform_utils
    from waymo_open_dataset.utils.frame_utils import \
        parse_range_image_and_camera_projection
except ImportError:
    raise ImportError(
        'Please clone "https://github.com/waymo-research/waymo-open-dataset.git" '
        'checkout branch "r1.3", and install the official devkit first')

import logging
import numpy as np
import os, sys, glob, pickle
import argparse
import tensorflow as tf
import matplotlib.image as mpimg

from pathlib import Path
from os.path import join, exists, dirname, abspath
from os import makedirs
from multiprocessing import Pool
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Waymo Dataset.')
    parser.add_argument('--dataset_path',
                        help='path to Waymo tfrecord files',
                        required=True)
    parser.add_argument(
        '--out_path',
        help='Output path to store pickle (default to dataet_path)',
        default=None,
        required=False)

    parser.add_argument('--workers',
                        help='Number of workers.',
                        default=16,
                        type=int)

    parser.add_argument('--split',
                        help='One of {train, val, test} (default train)',
                        default='train',
                        type=str)

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


class Waymo2KITTI():
    """Waymo to KITTI converter.

    This class converts tfrecord files from Waymo dataset to KITTI format.
    KITTI format : (type, truncated, occluded, alpha, bbox, dimensions(3), location(3), 
                   rotation_y(1), score(1, optional))
        type (string): Describes the type of object.
        truncated (float): Ranges from 0(non-truncated) to 1(truncated).
        occluded (int): Integer(0, 1, 2, 3) signifies state fully visible, partly
                        occluded, largely occluded, unknown.
        alpha (float): Observation angle of object, ranging [-pi..pi].
        bbox (float): 2d bounding box of object in the image.
        dimensions (float): 3D object dimensions: h, w, l in meters.
        location (float): 3D object location: x,y,z in camera coordinates (in meters).
        rotation_y (float): rotation around Y-axis in camera coordinates [-pi..pi].
        score (float): Only for predictions, indicating confidence in detection.
    
    Conversion writes following files:
        pointcloud(np.float32) : pointcloud data with shape [N, 6]. Consists of
                                 (x, y, z, intensity, elongation, timestamp).
        images(np.uint8): camera images are saved if `write_image` is True.
        calibrations(np.float32): Intinsic and Extrinsic matrix for all cameras.
        label(np.float32): Bounding box information in KITTI format.

    Args:
        dataset_path (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        workers (str): Number of workers for the parallel process.
        is_test (bool): Whether in the test_mode. Default: False.
    """

    def __init__(self, dataset_path, save_dir='', workers=8, split='train'):

        self.write_image = False
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = False

        self.classes = ['VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

        self.lidar_list = [
            '_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT'
        ]
        self.type_list = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

        self.selected_waymo_classes = self.classes

        self.selected_waymo_locations = None

        self.dataset_path = dataset_path
        self.save_dir = save_dir
        self.workers = int(workers)
        self.is_test = split == 'test'
        self.prefix = split + '_'
        self.save_track_id = False

        self.tfrecord_files = sorted(
            glob.glob(join(self.dataset_path, "*.tfrecord")))

        self.label_save_dir = f'{self.save_dir}/label_'
        self.label_all_save_dir = f'{self.save_dir}/label_all'
        self.image_save_dir = f'{self.save_dir}/image_'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.pose_save_dir = f'{self.save_dir}/pose'

        self.create_folder()

    def create_folder(self):
        if not self.is_test:
            dir_list1 = [
                self.label_all_save_dir, self.calib_save_dir,
                self.point_cloud_save_dir, self.pose_save_dir
            ]
            dir_list2 = [self.label_save_dir, self.image_save_dir]
        else:
            dir_list1 = [
                self.calib_save_dir, self.point_cloud_save_dir,
                self.pose_save_dir
            ]
            dir_list2 = [self.image_save_dir]
        for d in dir_list1:
            makedirs(d, exist_ok=True)
        for d in dir_list2:
            for i in range(5):
                makedirs(f'{d}{str(i)}', exist_ok=True)

    def convert(self):
        print(f"Start converting {len(self)} files ...")
        process_map(self.process_one,
                    range(len(self)),
                    max_workers=self.workers)

    def process_one(self, file_idx):
        print(f"Converting : {file_idx}")
        path = self.tfrecord_files[file_idx]
        dataset = tf.data.TFRecordDataset(path, compression_type='')

        for frame_idx, data in enumerate(dataset):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if (not frame.lasers[0].ri_return1.segmentation_label_compressed
               ) and (not self.is_test):
                continue

            if (self.selected_waymo_locations is not None and
                    frame.context.stats.location
                    not in self.selected_waymo_locations):
                continue

            if self.write_image:
                self.save_image(frame, file_idx, frame_idx)
            self.save_calib(frame, file_idx, frame_idx)
            self.save_lidar(frame, file_idx, frame_idx)
            self.save_pose(frame, file_idx, frame_idx)

    def __len__(self):
        return len(self.tfrecord_files)

    def save_image(self, frame, file_idx, frame_idx):
        for img in frame.images:
            img_path = Path(self.image_save_dir + str(img.name - 1)) / (
                self.prefix + str(file_idx).zfill(4) + str(frame_idx).zfill(4) +
                '.npy')
            image = tf.io.decode_jpeg(img.image).numpy()

            np.save(img_path, image)

    def save_calib(self, frame, file_idx, frame_idx):
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
        Tr_velo_to_cams = []
        calib_context = ''

        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                self.T_velo_to_front_cam = Tr_velo_to_cam.copy()
            Tr_velo_to_cam = Tr_velo_to_cam[:3, :].reshape((12,))
            Tr_velo_to_cams.append([f'{i:e}' for i in Tr_velo_to_cam])

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calib = list(camera_calib.reshape(12))
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        for i in range(5):
            calib_context += 'P' + str(i) + ': ' + \
                ' '.join(camera_calibs[i]) + '\n'
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
        for i in range(5):
            calib_context += 'Tr_velo_to_cam_' + str(i) + ': ' + \
                ' '.join(Tr_velo_to_cams[i]) + '\n'

        with open(
                f'{self.calib_save_dir}/{self.prefix}' +
                f'{str(file_idx).zfill(4)}{str(frame_idx).zfill(4)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)

    def save_pose(self, frame, file_idx, frame_idx):
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            join(f'{self.pose_save_dir}/{self.prefix}' +
                 f'{str(file_idx).zfill(4)}{str(frame_idx).zfill(4)}.txt'),
            pose)

    def save_lidar(self, frame, file_idx, frame_idx):
        range_images, camera_projections, seg_labels, range_image_top_pose = parse_range_image_and_camera_projection(
            frame)

        # First return
        points_0, cp_points_0, intensity_0, elongation_0, seg_label_0 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                seg_labels,
                camera_projections,
                range_image_top_pose,
                ri_index=0
            )
        points_0 = np.concatenate(points_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)
        elongation_0 = np.concatenate(elongation_0, axis=0)
        seg_label_0 = np.concatenate(seg_label_0, axis=0)

        # Second return
        points_1, cp_points_1, intensity_1, elongation_1, seg_label_1 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                seg_labels,
                camera_projections,
                range_image_top_pose,
                ri_index=1
            )
        points_1 = np.concatenate(points_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)
        elongation_1 = np.concatenate(elongation_1, axis=0)
        seg_label_1 = np.concatenate(seg_label_1, axis=0)

        points = np.concatenate([points_0, points_1], axis=0)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        elongation = np.concatenate([elongation_0, elongation_1], axis=0)
        semseg_labels = np.concatenate([seg_label_0, seg_label_1], axis=0)
        timestamp = frame.timestamp_micros * np.ones_like(intensity)

        # concatenate x,y,z, intensity, elongation, timestamp (6-dim)
        point_cloud = np.column_stack(
            (points, intensity, elongation, timestamp, semseg_labels))

        pc_path = f'{self.point_cloud_save_dir}/{self.prefix}' + \
            f'{str(file_idx).zfill(4)}{str(frame_idx).zfill(4)}.bin'
        point_cloud.astype(np.float32).tofile(pc_path)

    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           range_images,
                                           segmentation_labels,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        calibrations = sorted(frame.context.laser_calibrations,
                              key=lambda c: c.name)
        points = []
        cp_points = []
        intensity = []
        elongation = []
        semseg_labels = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = \
            transform_utils.get_rotation_matrix(
                range_image_top_pose_tensor[..., 0],
                range_image_top_pose_tensor[..., 1],
                range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = \
            range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant(
                        [c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            if self.filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(
                        value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data),
                                   cp.shape.dims)
            cp_points_tensor = tf.gather_nd(
                cp_tensor, tf.compat.v1.where(range_image_mask))
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                            tf.where(range_image_mask))
            intensity.append(intensity_tensor.numpy())

            elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                             tf.where(range_image_mask))
            elongation.append(elongation_tensor.numpy())

            if c.name in segmentation_labels:
                sl = segmentation_labels[c.name][ri_index]
                sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data),
                                       sl.shape.dims)
                sl_points_tensor = tf.gather_nd(sl_tensor,
                                                tf.where(range_image_mask))
            else:
                sl_points_tensor = tf.zeros([points_tensor.shape[0], 2],
                                            dtype=tf.int32)

            semseg_labels.append(sl_points_tensor.numpy())

        return points, cp_points, intensity, elongation, semseg_labels

    @staticmethod
    def cart_to_homo(mat):
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    args = parse_args()
    out_path = args.out_path
    if out_path is None:
        args.out_path = args.dataset_path
    if args.split not in ['train', 'val', 'test']:
        raise ValueError("split must be one of {train, val, test}")
    converter = Waymo2KITTI(args.dataset_path, args.out_path, args.workers,
                            args.split)
    converter.convert()
