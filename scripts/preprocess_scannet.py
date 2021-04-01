""" Preprocess ScanNet dataset and cache in Numpy data files """
import os
import struct
from pathlib import Path
from os.path import join
import concurrent.futures
import argparse
import json
import csv
import logging as log
import traceback
import zlib

import yaml
import numpy as np
from scipy.interpolate import RectBivariateSpline
import imageio
from tqdm import tqdm
import open3d as o3d
from open3d.core import Tensor
from open3d.ml.datasets import utils
PointCloud = o3d.t.geometry.PointCloud
BEVBox3D = utils.bev_box.BEVBox3D


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ScanNet Dataset.')
    parser.add_argument('--dataset_path',
                        help='path to ScanNet scans directory',
                        required=True)
    parser.add_argument('--out_path',
                        help='Output path to store processed data.',
                        default=None,
                        required=False)
    parser.add_argument(
        '--no_scene_pcd',
        help='Do not save scene point clouds and corresponding labels.',
        default=False,
        action='store_true',
        required=False)
    parser.add_argument(
        '--frame_pcd',
        help=
        'Extract individual camera frame point clouds and corresponding labels.',
        default=False,
        action='store_true',
        required=False)
    parser.add_argument(
        '--frame_color',
        help='Use frame RGB data to produce colored point clouds.',
        default=False,
        action='store_true',
        required=False)
    parser.add_argument('--frame_skip',
                        help='Only process one in frame_skip frames',
                        default=10,
                        type=int,
                        required=False)
    parser.add_argument(
        '--only_stats',
        help='Do not preprocess. Only compute dataset statistics.',
        default=False,
        action='store_true',
        required=False)

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def represents_int(s):
    """Judge whether string s represents an int.
    Args:
        s(str): The input string to be judged.
    Returns:
        bool: Whether s represents int or not.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


class RGBDFrame():
    """Reading a single RGBD frame with metadata from a .sens file"""

    def __init__(self, file_handle, skip=False):
        """
        Args:
            file_handle: Open file handle for <scanId>.sens file at the start of
                the next frame.
            skip (bool): Do not read this frame
        """
        # from current frame to base frame
        self.camera_to_world = np.asarray(struct.unpack(
            'f' * 16, file_handle.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        # timestamp is in microseconds
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        if skip:
            self.color_data = None
            self.depth_data = None
            file_handle.seek(self.color_size_bytes + self.depth_size_bytes,
                             os.SEEK_CUR)
        else:
            self.color_data = b''.join(
                struct.unpack('c' * self.color_size_bytes,
                              file_handle.read(self.color_size_bytes)))
            self.depth_data = b''.join(
                struct.unpack('c' * self.depth_size_bytes,
                              file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        else:
            raise RuntimeError("Unknown depth compression type " +
                               compression_type)

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        else:
            raise RuntimeError("Unknown color compression type " +
                               compression_type)

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


class SensorData:
    """Reader for .sens files. Reads depth and color frames and associated
    metadata. This creates an iterable object."""

    COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
    COMPRESSION_TYPE_DEPTH = {
        -1: 'unknown',
        0: 'raw_ushort',
        1: 'zlib_ushort',
        2: 'occi_ushort'
    }

    def __init__(self, filename, frame_skip=1):
        self.frame_skip = frame_skip
        self.version = 4
        self.file_handle = open(filename, 'rb')
        version = struct.unpack('I', self.file_handle.read(4))[0]
        assert self.version == version, (
            f"sens file is version {version} but" +
            " reader code is version {self.version}")
        strlen = struct.unpack('Q', self.file_handle.read(8))[0]
        self.sensor_name = b''.join(
            struct.unpack('c' * strlen,
                          self.file_handle.read(strlen))).decode("utf-8")
        self.intrinsic_color = np.asarray(struct.unpack(
            'f' * 16, self.file_handle.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        self.extrinsic_color = np.asarray(struct.unpack(
            'f' * 16, self.file_handle.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        self.intrinsic_depth = np.asarray(struct.unpack(
            'f' * 16, self.file_handle.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        self.extrinsic_depth = np.asarray(struct.unpack(
            'f' * 16, self.file_handle.read(16 * 4)),
                                          dtype=np.float32).reshape(4, 4)
        self.color_compression_type = self.COMPRESSION_TYPE_COLOR[struct.unpack(
            'i', self.file_handle.read(4))[0]]
        self.depth_compression_type = self.COMPRESSION_TYPE_DEPTH[struct.unpack(
            'i', self.file_handle.read(4))[0]]
        self.color_width = struct.unpack('I', self.file_handle.read(4))[0]
        self.color_height = struct.unpack('I', self.file_handle.read(4))[0]
        self.depth_width = struct.unpack('I', self.file_handle.read(4))[0]
        self.depth_height = struct.unpack('I', self.file_handle.read(4))[0]
        self.depth_shift = struct.unpack('f', self.file_handle.read(4))[0]
        self.num_frames = struct.unpack('Q', self.file_handle.read(8))[0]

        self.depth_max = 3.5  # Occipital ST01 recommended range in m

    def __iter__(self):
        for frame_id in range(self.num_frames):
            skip = frame_id % self.frame_skip != 0
            frame = RGBDFrame(self.file_handle, skip=skip)
            if not skip:
                depth_data = frame.decompress_depth(self.depth_compression_type)
                depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                    self.depth_height, self.depth_width)
                color = frame.decompress_color(self.color_compression_type)
                yield (depth, color, frame.camera_to_world)


class ScannetProcess():
    """Preprocess ScanNet.
    This class converts ScanNet raw data into npy files.
    Args:
        dataset_path (str): Directory to load ScanNet data.
        out_path (str): Directory to save pickle file(infos).
        max_num_point (int): max vertices per scene point cloud
        scene_pcd (bool): Save scene pointclouds?
        frame_pcd (bool): Save frame pointclouds?
        frame_color (bool): Save color for frame pointclouds?
        frame_skip (int): Save only one in `frame_skip` frames. Default 100. Set
            to 1 to save all frames.
        process_id (int,int): tuple of (process_id, total_processes). The list
            of scans can be split up and this instance will only process scans
            corresponding to process_id
    """

    DONOTCARE_IDS = np.array([])
    OBJ_CLASS_IDS = np.array(
        [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

    def __init__(self,
                 dataset_path,
                 out_path,
                 max_num_point=100000,
                 scene_pcd=True,
                 frame_pcd=False,
                 frame_color=False,
                 frame_skip=100,
                 process_id=(0, 1)):

        self.out_path = out_path
        self.out_frame_path = join(out_path, "frames")
        if frame_pcd:
            os.makedirs(self.out_frame_path, exist_ok=True)
        else:
            os.makedirs(self.out_path, exist_ok=True)
        self.dataset_path = dataset_path
        self.max_num_point = max_num_point
        self.scene_pcd = scene_pcd
        self.frame_pcd = frame_pcd
        self.min_instance_pts = 10

        # Parallelize tasks that do not need the GIL
        self.max_workers = 4
        self._runner = None

        scans = os.listdir(dataset_path)
        self.scans = []
        for scan in scans[process_id[0]::process_id[1]]:
            name = scan.split('/')[-1]
            if 'scene' in name and len(name) == 12:
                self.scans.append(scan)

        self.frame_skip = frame_skip
        self.frame_color = frame_color
        self.pcd_stride = 1  # downsample frames before pcd conversion
        self.process_id = process_id

        log.info(f"Total number of scans : {len(self.scans)}")

    def convert(self):
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers) as self._runner:
            errors = []
            for scan in tqdm(
                    self.scans,
                    desc=f"{self.process_id[0]}/{self.process_id[1]}:convert",
                    unit="scan"):
                try:
                    self.process_scene(scan)
                    log.info("Pending tasks: %d",
                             self._runner._work_queue.qsize())
                except FileNotFoundError:
                    errmsg = f'{scan}: ' + traceback.format_exc()
                    log.warning(errmsg)
                    errors.append(errmsg)

            if errors:
                with open(join(self.out_path, 'errors.txt'), 'w') as errfile:
                    errfile.write("Processing failed:\n" + "\n".join(errors))

    def process_scene(self, scan):
        # if (isfile(f'{join(self.out_path, scan)}_vert.npy') and
        #         isfile(f'{join(self.out_path, scan)}_sem_label.npy') and
        #         isfile(f'{join(self.out_path, scan)}_ins_label.npy') and
        #         isfile(f'{join(self.out_path, scan)}_bbox.npy')):
        #     return

        log.info("Processing " + scan)
        in_path = join(self.dataset_path, scan)
        mesh_file = join(in_path, scan + '_vh_clean_2.ply')
        agg_file = join(in_path, scan + '.aggregation.json')
        seg_file = join(in_path, scan + '_vh_clean_2.0.010000.segs.json')

        meta_file = join(in_path, scan + '.txt')
        label_map_file = str(
            Path(__file__).parent /
            '../ml3d/datasets/_resources/scannet/scannetv2-labels.combined.tsv')
        (mesh_vertices, semantic_labels, instance_labels, instance_bboxes,
         instance2semantic, axis_align_matrix,
         color_to_depth) = self.export(mesh_file, agg_file, seg_file, meta_file,
                                       label_map_file)

        mask = np.logical_not(np.in1d(semantic_labels, self.DONOTCARE_IDS))
        mesh_vertices = mesh_vertices[mask, :]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        num_instances = len(np.unique(instance_labels))
        log.info(f'Num of instances: {num_instances}')

        bbox_mask = np.in1d(instance_bboxes[:, -1], self.OBJ_CLASS_IDS)
        instance_bboxes = instance_bboxes[bbox_mask, :]
        log.info(f'Num of care instances: {instance_bboxes.shape[0]}')

        N = mesh_vertices.shape[0]
        if N > self.max_num_point:
            choices = np.random.choice(N, self.max_num_point, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            semantic_labels = semantic_labels[choices]
            instance_labels = instance_labels[choices]

        if self.frame_pcd:
            self.export_frame_pointclouds(scan, instance_bboxes,
                                          axis_align_matrix, color_to_depth)

        if self.scene_pcd:
            self._runner.submit(np.save,
                                f'{join(self.out_path, scan)}_vert.npy',
                                mesh_vertices)
            self._runner.submit(np.save,
                                f'{join(self.out_path, scan)}_sem_label.npy',
                                semantic_labels)
            self._runner.submit(np.save,
                                f'{join(self.out_path, scan)}_ins_label.npy',
                                instance_labels)
            self._runner.submit(np.save,
                                f'{join(self.out_path, scan)}_bbox.npy',
                                instance_bboxes)

    @staticmethod
    def rgbd_to_pointcloud(color,
                           depth,
                           color_intrinsic,
                           depth_intrinsic,
                           color_to_depth,
                           depth_scale=1000.0,
                           depth_max=3.5,
                           stride=1):
        """
        Convert unaligned color and depth image pair to a point cloud. If you
        have an aligned color and depth pair, use
        PointCloud.create_from_rgbd_image() directly.

        Args:
            color ((rows, cols, 3) numpy array or open3d.t.geometry.Image):
                color image. Color values are scaled down by 255 if this is uint8
                If this is None, an all zeros "colors" attribute is returned
            depth (open3d.t.geometry.Image): 1 channel depth image
            color_intrinsic ((3,3) open3d.core.Tensor): intrinsic matrix for
                color camera
            depth_intrinsic ((3,3) open3d.core.Tensor): intrinsic matrix for
                depth camera
            color_to_depth ((4,4) open3d.core.Tensor): extrinsic homogenous
                transform with x_depth = color_to_depth @ x_color
           depth_scale (optional, default=1000.0): depth (m) = depth_value /
                   depth_scale
           depth_max (optional, default=3.5): Point cloud is truncated beyond
                    depth_max (m)
           stride (optional, default=1): subsample factor for depth image

        Returns: PointCloud with "points" and "colors" attributes
        """
        pcd = PointCloud.create_from_depth_image(depth, depth_intrinsic,
                                                 color_to_depth, depth_scale,
                                                 depth_max, stride)
        pcd.point["points"] = pcd.point["points"].cpu()
        pts = pcd.point["points"]
        if not color:
            pcd.point["colors"] = Tensor.zeros(pts.shape)
            return pcd
        if isinstance(color, o3d.t.geometry.Image):
            color = color.as_Tensor().cpu().numpy()
        im_coords = [None, None]
        for idx in range(2):  # x,y ->row,ch
            im_coords[1 - idx] = np.where(
                (pts[:, 2].abs() > 1e-6).numpy(),
                (pts[:, idx] / pts[:, 2] * color_intrinsic[idx, idx] +
                 color_intrinsic[idx, 2]).numpy(), -1)
            im_coords[1 - idx] = np.where(
                np.logical_and(0 <= im_coords[1 - idx],
                               im_coords[1 - idx] <= color.shape[1 - idx] - 0),
                im_coords[1 - idx], -1)

        pts_colors = np.empty(pts.shape, dtype=np.float32)
        color_scale = (np.float32(1. / 255)
                       if color.dtype == np.uint8 else np.float32(1.0))
        for channel in range(3):
            colors_interp = RectBivariateSpline(np.arange(color.shape[0]),
                                                np.arange(color.shape[1]),
                                                color[:, :, channel],
                                                kx=1,
                                                ky=1)
            pts_colors[:, channel] = np.where(
                np.logical_and(im_coords[0] >= 0, im_coords[1] >= 0),
                color_scale *
                colors_interp(im_coords[0], im_coords[1], grid=False), 0)
        pcd.point["colors"] = Tensor.from_numpy(pts_colors)
        return pcd

    def export_frame_pointclouds(self, scan, instance_bboxes, axis_align_matrix,
                                 color_to_depth):
        """Read <scan>/<scan>.sens file for a scene and convert depth frames to
        point clouds.  Also estimate instance bounding boxes inside the frames.
        Save point clouds and instance bboxes in
        <scanId>_<frameID>_{vert,bbox}.npy files

        Args:
            scan (str): scanId
            instance_bboxes (Iterable): All bboxes in the scene (excluding
            dont_care)
            axis_align_matrix ((4,4) array): world to label (bounding boxes)
                frame transform  (from <scanId>.txt file)
            color_to_depth ((4,4) array): color camera to depth camera extrinsic
                transform for the scene (from <scanId>.txt file)

        """
        # Assume color ref == camera ref
        # label_to_depth = color_to_depth @ np.linalg.inv( axis_align_matrix @ camera_to_world)
        device = o3d.core.Device(
            'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0')
        frame_id = 0
        n_frames = 0  # frames with visible bboxes
        n_bbox = 0  # visible bboxes
        sens_file = join(self.dataset_path, scan, scan + '.sens')
        sensor_data = SensorData(sens_file, self.frame_skip)
        intrinsic_depth = o3d.core.Tensor(sensor_data.intrinsic_depth[:3, :3],
                                          device=device)
        intrinsic_color = o3d.core.Tensor(sensor_data.intrinsic_color[:3, :3],
                                          device=device)

        pcd_future = [None, None]
        label_to_depth = [None, None]
        for (depth, color, camera_to_world) in tqdm(sensor_data,
                                                    desc=scan,
                                                    unit='frame'):
            if not self.frame_color:
                color = None
            frame_instance_bboxes = []
            depth_image = o3d.t.geometry.Image(
                o3d.core.Tensor.from_numpy(depth).to(device))
            # Assume color ref == camera ref
            label_to_depth[1] = color_to_depth @ np.linalg.inv(
                axis_align_matrix @ camera_to_world)
            pcd_future[1] = self._runner.submit(
                self.rgbd_to_pointcloud, color, depth_image, intrinsic_color,
                intrinsic_depth,
                Tensor.eye(4, device=device, dtype=o3d.core.Dtype.Float32),
                sensor_data.depth_shift, sensor_data.depth_max, self.pcd_stride)

            if not pcd_future[0]:
                pcd_future[0] = pcd_future[1]
                label_to_depth[0] = label_to_depth[1]
                continue
            mesh_vertices = np.hstack(
                (pcd_future[0].result().point["points"].cpu().numpy(),
                 pcd_future[0].result().point["colors"].cpu().numpy()))
            for bbox in instance_bboxes:
                bev_bbox = BEVBox3D(
                    bbox[:3],  # center xyz
                    bbox[3:6],  # size w=dx, l=dy, h=dz
                    0,  # yaw angle
                    bbox[6],  # label
                    1,  # confidence
                    world_cam=label_to_depth[0].T,
                    cam_img=sensor_data.intrinsic_depth.T)
                bev_bbox_cam = bev_bbox.to_camera_bev()
                if len(bev_bbox_cam.inside(
                        mesh_vertices[:, :3])) > self.min_instance_pts:
                    frame_instance_bboxes.append(bev_bbox_cam)
            # Don't save empty frames
            if frame_instance_bboxes:
                n_bbox += len(frame_instance_bboxes)
                n_frames += 1
                frame_pcd = f"{scan}_{frame_id:06}"
                np_bboxes = np.vstack(
                    list(box.to_xyzwlhyc() for box in frame_instance_bboxes))
                self._runner.submit(
                    np.save, f'{join(self.out_frame_path, frame_pcd)}_bbox.npy',
                    np_bboxes)
                self._runner.submit(
                    np.save, f'{join(self.out_frame_path, frame_pcd)}_vert.npy',
                    mesh_vertices)
            pcd_future[0] = pcd_future[1]
            label_to_depth[0] = label_to_depth[1]
            frame_id += self.frame_skip

        # Save last frame
        if frame_instance_bboxes:
            mesh_vertices = np.hstack(
                (pcd_future[0].result().point["points"].cpu().numpy(),
                 pcd_future[0].result().point["colors"].cpu().numpy()))
            frame_pcd = f"{scan}_{frame_id:06}"
            self._runner.submit(
                np.save, f'{join(self.out_frame_path, frame_pcd)}_vert.npy',
                mesh_vertices)
            self._runner.submit(
                np.save, f'{join(self.out_frame_path, frame_pcd)}_bbox.npy',
                frame_instance_bboxes)
        log.info(f"{n_bbox} instances in {n_frames} frames")

    def export(self, mesh_file, agg_file, seg_file, meta_file, label_map_file):
        mesh_vertices = self.read_mesh_vertices_rgb(mesh_file)
        label_map = self.read_label_mapping(label_map_file,
                                            label_from='raw_category',
                                            label_to='nyu40id')

        # Load axis alignment matrix
        lines = open(meta_file).readlines()
        read_lines = 0
        for line in lines:
            if read_lines == 2:
                break
            if 'axisAlignment' in line:
                axis_align_matrix = np.fromstring(
                    line.strip('axisAlignment = '), count=16, sep=' ').reshape(
                        (4, 4))
                read_lines += 1
            elif 'colorToDepthExtrinsics' in line:
                color_to_depth = np.fromstring(
                    line.strip('colorToDepthExtrinsics = '), count=16,
                    sep=' ').reshape((4, 4))
                read_lines += 1
        pts = np.ones((mesh_vertices.shape[0], 4))
        pts[:, 0:3] = mesh_vertices[:, 0:3]
        pts = np.dot(pts, axis_align_matrix.transpose())
        mesh_vertices[:, 0:3] = pts[:, 0:3]

        # Load instance and semantic labels.
        object_id_to_segs, label_to_segs = self.read_aggregation(agg_file)
        seg_to_verts, num_verts = self.read_segmentation(seg_file)

        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
        object_id_to_label_id = {}
        for label, segs in label_to_segs.items():
            label_id = label_map[label]
            for seg in segs:
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id

        instance_ids = np.zeros(shape=(num_verts),
                                dtype=np.uint32)  # 0: unannotated
        num_instances = len(np.unique(list(object_id_to_segs.keys())))
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][0]

        instance_bboxes = np.zeros((num_instances, 7))
        for obj_id in object_id_to_segs:
            label_id = object_id_to_label_id[obj_id]
            obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
            if len(obj_pc) == 0:
                continue
            xmin = np.min(obj_pc[:, 0])
            ymin = np.min(obj_pc[:, 1])
            zmin = np.min(obj_pc[:, 2])
            xmax = np.max(obj_pc[:, 0])
            ymax = np.max(obj_pc[:, 1])
            zmax = np.max(obj_pc[:, 2])
            bbox = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2,
                             (zmin + zmax) / 2, xmax - xmin, ymax - ymin,
                             zmax - zmin, label_id])
            # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
            instance_bboxes[obj_id - 1, :] = bbox

        return (mesh_vertices, label_ids, instance_ids, instance_bboxes,
                object_id_to_label_id, axis_align_matrix, color_to_depth)

    @staticmethod
    def read_label_mapping(filename,
                           label_from='raw_category',
                           label_to='nyu40id'):
        assert os.path.isfile(filename)
        mapping = dict()
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                mapping[row[label_from]] = int(row[label_to])
        if represents_int(list(mapping.keys())[0]):
            mapping = {int(k): v for k, v in mapping.items()}
        return mapping

    @staticmethod
    def read_mesh_vertices_rgb(filename):
        """Read XYZ and RGB for each vertex.
        Args:
            filename(str): The name of the mesh vertices file.
        Returns:
            Vertices. Note that RGB values are in 0-255.
        """
        assert os.path.isfile(filename)
        with open(filename, 'rb') as f:
            data = o3d.t.io.read_point_cloud(f.name).point
            points = data["points"].numpy().astype(np.float32)
            colors = data["colors"].numpy().astype(np.float32)
            vertices = np.concatenate([points, colors], axis=1)

        return vertices

    @staticmethod
    def read_aggregation(filename):
        assert os.path.isfile(filename)
        object_id_to_segs = {}
        label_to_segs = {}
        with open(filename) as f:
            data = json.load(f)
            num_objects = len(data['segGroups'])
            for i in range(num_objects):
                object_id = data['segGroups'][i][
                    'objectId'] + 1  # instance ids should be 1-indexed
                label = data['segGroups'][i]['label']
                segs = data['segGroups'][i]['segments']
                object_id_to_segs[object_id] = segs
                if label in label_to_segs:
                    label_to_segs[label].extend(segs)
                else:
                    label_to_segs[label] = segs
        return object_id_to_segs, label_to_segs

    @staticmethod
    def read_segmentation(filename):
        assert os.path.isfile(filename)
        seg_to_verts = {}
        with open(filename) as f:
            data = json.load(f)
            num_verts = len(data['segIndices'])
            for i in range(num_verts):
                seg_id = data['segIndices'][i]
                if seg_id in seg_to_verts:
                    seg_to_verts[seg_id].append(i)
                else:
                    seg_to_verts[seg_id] = [i]
        return seg_to_verts, num_verts

    def compute_dataset_statistics(self):

        def get_scene_stats(scan):
            try:
                mesh_vertices = np.load(f'{join(self.out_path, scan)}_vert.npy')
                semantic_labels = np.load(
                    f'{join(self.out_path, scan)}_sem_label.npy')
                instance_labels = np.load(
                    f'{join(self.out_path, scan)}_ins_label.npy')
                instance_bboxes = np.load(
                    f'{join(self.out_path, scan)}_bbox.npy')
            except FileNotFoundError:
                log.warning(
                    f"Some files are missing: {join(self.out_path, scan)}_*.npy."
                    + " Please re-run preprocessing.")
                return None
            return utils.statistics.compute_scene_stats(mesh_vertices,
                                                        semantic_labels,
                                                        instance_labels,
                                                        instance_bboxes)

        def get_frame_stats(frame):
            try:
                mesh_vertices = np.load(
                    f'{join(self.out_frame_path, frame)}_vert.npy')
                instance_bboxes = np.load(
                    f'{join(self.out_frame_path, frame)}_bbox.npy')
            except FileNotFoundError:
                log.warning(
                    f"Some files are missing: {join(self.out_path, frame)}_*.npy."
                    + " Please re-run preprocessing.")
                return None
            return utils.statistics.compute_scene_stats(mesh_vertices, None,
                                                        None, instance_bboxes)

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers) as runner:
            if self.scene_pcd:
                scene_stats = list(
                    tqdm(runner.map(get_scene_stats, self.scans),
                         total=len(self.scans),
                         desc="scene_stats",
                         unit="scene"))

                dataset_stats = utils.statistics.compute_dataset_stats(
                    scene_stats)
                with open(join(self.out_path, 'scene_summary.yaml'),
                          'w') as sumfile:
                    yaml.dump(dataset_stats, sumfile)
            # if self.frame_pcd:
            #     frame_stats = list(
            #         tqdm(runner.map(get_frame_stats, self.scans),
            #              total=len(self.scans),
            #              desc="frame_stats",
            #              unit="frame"))

            #     dataset_stats = utils.statistics.compute_dataset_stats(
            #         frame_stats)
            #     with open(join(self.out_path, 'frame_summary.yaml'),
            #               'w') as sumfile:
            #         yaml.dump(dataset_stats, sumfile)


if __name__ == '__main__':

    log.basicConfig(level=log.INFO)
    args = parse_args()
    if args.out_path is None:
        args.out_path = args.dataset_path

    max_processes = 3
    if not args.only_stats:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_processes) as executor:
            converter = [
                ScannetProcess(args.dataset_path,
                               args.out_path,
                               scene_pcd=not args.no_scene_pcd,
                               frame_pcd=args.frame_pcd,
                               frame_color=args.frame_color,
                               frame_skip=args.frame_skip,
                               process_id=(process, max_processes))
                for process in range(max_processes)
            ]
            futures = [
                executor.submit(converter[process].convert)
                for process in range(max_processes)
            ]
            for process in range(max_processes):
                futures[process].result()

    ScannetProcess(args.dataset_path,
                   args.out_path,
                   scene_pcd=not args.no_scene_pcd,
                   frame_pcd=args.frame_pcd,
                   frame_color=args.frame_color,
                   frame_skip=args.frame_skip).compute_dataset_statistics()
