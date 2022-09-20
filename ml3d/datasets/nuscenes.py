import os
import pickle
from os.path import join
from pathlib import Path
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base_dataset import BaseDataset
from ..utils import DATASET
from .utils import BEVBox3D
import open3d as o3d

log = logging.getLogger(__name__)


class NuScenes(BaseDataset):
    """This class is used to create a dataset based on the NuScenes 3D dataset,
    and used in object detection, visualizer, training, or testing.

    The NuScenes 3D dataset is best suited for autonomous driving applications.
    """

    def __init__(self,
                 dataset_path,
                 info_path=None,
                 name='NuScenes',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            info_path: The path to the file that includes information about the
                dataset. This is default to dataset path if nothing is provided.
            name: The name of the dataset (NuScenes in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.

        Returns:
            class: The corresponding class.
        """
        if info_path is None:
            info_path = dataset_path

        super().__init__(dataset_path=dataset_path,
                         info_path=info_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 10
        self.label_to_names = self.get_label_to_names()

        self.train_info = {}
        self.test_info = {}
        self.val_info = {}

        if os.path.exists(join(info_path, 'infos_train.pkl')):
            self.train_info = pickle.load(
                open(join(info_path, 'infos_train.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'infos_val.pkl')):
            self.val_info = pickle.load(
                open(join(info_path, 'infos_val.pkl'), 'rb'))

        if os.path.exists(join(info_path, 'infos_test.pkl')):
            self.test_info = pickle.load(
                open(join(info_path, 'infos_test.pkl'), 'rb'))

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'ignore',
            1: 'barrier',
            2: 'bicycle',
            3: 'bus',
            4: 'car',
            5: 'construction_vehicle',
            6: 'motorcycle',
            7: 'pedestrian',
            8: 'traffic_cone',
            9: 'trailer',
            10: 'truck'
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()

        return np.fromfile(path, dtype=np.float32).reshape(-1, 5)

    @staticmethod
    def read_label(info, calib):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        mask = info['num_lidar_pts'] != 0
        boxes = info['gt_boxes'][mask]
        names = info['gt_names'][mask]

        objects = []
        for name, box in zip(names, boxes):

            center = [float(box[0]), float(box[1]), float(box[2])]
            size = [float(box[3]), float(box[5]), float(box[4])]
            ry = float(box[6])

            yaw = ry - np.pi
            yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi

            world_cam = calib['world_cam']

            objects.append(BEVBox3D(center, size, yaw, name, -1.0, world_cam))
            objects[-1].yaw = ry

        return objects

    def read_cams(self, cam_dict):
        """Reads image data from the cam dict provided.

        Args:
            cam_dict (Dict): Mapping from camera names to dict with image
                information ('data_path', 'sensor2lidar_translation',
                'sensor2lidar_rotation', 'cam_intrinsic').

        Returns:
            A dict with keys as camera names and value as images.
        """
        assert [Path(val['data_path']).exists() for _, val in cam_dict.items()]

        res_dict = dict()
        for cam in cam_dict.keys():
            res_dict[cam] = dict()
            res_dict[cam]['img'] = np.array(
                o3d.io.read_image(cam_dict[cam]['data_path']))

            # obtain lidar to cam transformation matrix
            lidar2cam_r = np.linalg.inv(cam_dict[cam]['sensor2lidar_rotation'])
            lidar2cam_t = cam_dict[cam][
                'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_dict[cam]['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            # obtain lidar to image transformation matrix
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)

            res_dict[cam]['lidar2cam_rt'] = lidar2cam_rt
            res_dict[cam]['lidar2img_rt'] = lidar2img_rt
            res_dict[cam]['cam_intrinsic'] = cam_dict[cam]['cam_intrinsic']

        return res_dict

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return NuSceneSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
                split name should be one of 'training', 'test', 'validation', or
                'all'.
        """
        if split in ['train', 'training']:
            return self.train_info
        elif split in ['test', 'testing']:
            return self.test_info
        elif split in ['val', 'validation']:
            return self.val_info

        raise ValueError("Invalid split {}".format(split))

    def is_tested():
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
                        attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        pass

    def save_test_result():
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
                attribute passed.
            attr: The attributes that correspond to the outputs passed in
                results.
        """
        pass


class NuSceneSplit():

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg

        self.infos = dataset.get_split_list(split)
        self.path_list = []
        for info in self.infos:
            self.path_list.append(info['lidar_path'])

        log.info("Found {} pointclouds for {}".format(len(self.infos), split))

        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.infos)

    def get_data(self, idx):
        info = self.infos[idx]
        lidar_path = info['lidar_path']

        world_cam = np.eye(4)
        world_cam[:3, :3] = R.from_quat(info['lidar2ego_rot']).as_matrix()
        world_cam[:3, -1] = info['lidar2ego_tr']
        calib = {'world_cam': world_cam.T}

        pc = self.dataset.read_lidar(lidar_path)
        label = self.dataset.read_label(info, calib)

        data = {
            'point': pc,
            'feat': None,
            'calib': calib,
            'bounding_boxes': label,
        }

        if 'cams' in info:
            data['cams'] = self.dataset.read_cams(info['cams'])

        return data

    def get_attr(self, idx):
        info = self.infos[idx]
        pc_path = info['lidar_path']
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(NuScenes)
