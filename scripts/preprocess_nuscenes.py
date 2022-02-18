try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
except ImportError:
    raise ImportError('Please run "pip install nuscenes-devkit" '
                      'to install the official devkit first.')

import logging
import os
import pickle
from os.path import join
from os import makedirs
import argparse
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess NuScenes Dataset.')
    parser.add_argument('--dataset_path',
                        help='path to Nuscene root',
                        required=True)
    parser.add_argument(
        '--out_path',
        help='Output path to store pickle (default to dataet_path)',
        default=None,
        required=False)

    parser.add_argument('--version',
                        help='one of {v1.0-trainval, v1.0-test, v1.0-mini}',
                        default='v1.0-trainval')

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


class NuScenesProcess():
    """Preprocess NuScenes.

    This class collects paths and labels using nuscenes-devkit.

    Args:
        dataset_path (str): Directory to load nuscenes data.
        out_path (str): Directory to save pickle file(infos).
        version (str): version of dataset. Default: v1.0-trainval.
    """

    def __init__(self, dataset_path, out_path, version='v1.0-trainval'):

        assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
        self.is_test = 'test' in version
        self.out_path = out_path

        self.nusc = NuScenes(version=version,
                             dataroot=dataset_path,
                             verbose=True)

        if version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise ValueError('unknown')

        self.version = version
        self.mapping = self.get_mapping()

        available_scenes = self.get_available_scenes()
        names = [sc['name'] for sc in available_scenes]
        train_scenes = list(filter(lambda x: x in names, train_scenes))
        val_scenes = list(filter(lambda x: x in names, val_scenes))

        train_scenes = set(
            [available_scenes[names.index(s)]['token'] for s in train_scenes])
        val_scenes = set(
            [available_scenes[names.index(s)]['token'] for s in val_scenes])

        if not self.is_test:
            print(
                f"train_scenes : {len(train_scenes)}, val_scenes : {len(val_scenes)}"
            )
        else:
            print(f"test scenes : {len(train_scenes)}")

        self.train_scenes = train_scenes
        self.val_scenes = val_scenes

    @staticmethod
    def get_mapping():
        mapping = {
            'movable_object.barrier': 'barrier',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }
        return mapping

    def convert(self):
        train_info, val_info = self.process_scenes()
        out_path = self.out_path
        makedirs(out_path, exist_ok=True)

        if self.is_test:
            with open(join(out_path, 'infos_test.pkl'), 'wb') as f:
                pickle.dump(train_info, f)
            print(f"Saved test info at {join(out_path, 'infos_test.pkl')}")
        else:
            with open(join(out_path, 'infos_train.pkl'), 'wb') as f:
                pickle.dump(train_info, f)
            with open(join(out_path, 'infos_val.pkl'), 'wb') as f:
                pickle.dump(val_info, f)
            print(f"Saved train info at {join(out_path, 'infos_train.pkl')}")
            print(f"Saved val info at {join(out_path, 'infos_val.pkl')}")

    def obtain_sensor2top(self,
                          sensor_token,
                          l2e_t,
                          l2e_r_mat,
                          e2g_t,
                          e2g_r_mat,
                          sensor_type='lidar'):
        """Obtain the info with RT matric from general sensor to Top LiDAR.

        Args:
            nusc (class): Dataset class in the nuScenes dataset.
            sensor_token (str): Sample data token corresponding to the
                specific sensor type.
            l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
            l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
                in shape (3, 3).
            e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
            e2g_r_mat (np.ndarray): Rotation matrix from ego to global
                in shape (3, 3).
            sensor_type (str): Sensor to calibrate. Default: 'lidar'.

        Returns:
            sweep (dict): Sweep information after transformation.
        """
        nusc = self.nusc
        sd_rec = nusc.get('sample_data', sensor_token)
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        data_path = str(nusc.get_sample_data_path(sd_rec['token']))
        if os.getcwd() in data_path:  # path from lyftdataset is absolute path
            data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
        sweep = {
            'data_path': data_path,
            'type': sensor_type,
            'sample_data_token': sd_rec['token'],
            'sensor2ego_translation': cs_record['translation'],
            'sensor2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sd_rec['timestamp']
        }
        l2e_r_s = sweep['sensor2ego_rotation']
        l2e_t_s = sweep['sensor2ego_translation']
        e2g_r_s = sweep['ego2global_rotation']
        e2g_t_s = sweep['ego2global_translation']

        # obtain the RT from sensor to Top LiDAR
        # sweep->ego->global->ego'->lidar
        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                     ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
        sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
        sweep['sensor2lidar_translation'] = T
        return sweep

    def process_scenes(self):
        nusc = self.nusc
        train_info = []
        val_info = []

        for sample in tqdm(nusc.sample):
            lidar_token = sample['data']['LIDAR_TOP']
            sd_rec = nusc.get('sample_data', lidar_token)
            calib_rec = nusc.get('calibrated_sensor',
                                 sd_rec['calibrated_sensor_token'])
            pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])

            lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

            lidar_path = os.path.abspath(lidar_path)
            assert os.path.exists(lidar_path)

            data = {
                'lidar_path': lidar_path,
                'token': sample['token'],
                'cams': dict(),
                'lidar2ego_tr': calib_rec['translation'],
                'lidar2ego_rot': calib_rec['rotation'],
                'ego2global_tr': pose_rec['translation'],
                'ego2global_rot': pose_rec['rotation'],
                'timestamp': sample['timestamp']
            }

            l2e_r = data['lidar2ego_rot']
            l2e_t = data['lidar2ego_tr']
            e2g_r = data['ego2global_rot']
            e2g_t = data['ego2global_tr']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 6 image's information per frame
            camera_types = [
                'CAM_FRONT',
                'CAM_FRONT_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
            ]
            for cam in camera_types:
                cam_token = sample['data'][cam]
                cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
                cam_info = self.obtain_sensor2top(cam_token, l2e_t, l2e_r_mat,
                                                  e2g_t, e2g_r_mat, cam)
                cam_info.update(cam_intrinsic=cam_intrinsic)
                data['cams'].update({cam: cam_info})

            if not self.is_test:
                annotations = [
                    nusc.get('sample_annotation', token)
                    for token in sample['anns']
                ]
                locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                rots = np.array([
                    b.orientation.yaw_pitch_roll[0] for b in boxes
                ]).reshape(-1, 1)

                valid_flag = np.array([
                    (ann['num_lidar_pts'] + ann['num_radar_pts']) > 0
                    for ann in annotations
                ]).reshape(-1)

                names = [
                    'ignore'
                    if b.name not in self.mapping else self.mapping[b.name]
                    for b in boxes
                ]
                names = np.array(names)

                gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2],
                                          axis=1)

                data['gt_boxes'] = gt_boxes
                data['gt_names'] = names
                data['num_lidar_pts'] = np.array(
                    [ann['num_lidar_pts'] for ann in annotations])
                data['valid_flag'] = valid_flag

            if sample['scene_token'] in self.train_scenes:
                train_info.append(data)
            else:
                val_info.append(data)

        return train_info, val_info

    def get_available_scenes(self):
        nusc = self.nusc
        available_scenes = []

        for scene in nusc.scene:
            token = scene['token']
            scene_rec = nusc.get('scene', token)
            sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
            sample_data = nusc.get('sample_data',
                                   sample_rec['data']['LIDAR_TOP'])

            lidar_path, boxes, _ = nusc.get_sample_data(sample_data['token'])
            lidar_path = str(lidar_path)

            if not os.path.exists(lidar_path):
                continue
            else:
                available_scenes.append(scene)

        return available_scenes


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    args = parse_args()
    out_path = args.out_path
    if out_path is None:
        args.out_path = args.dataset_path
    converter = NuScenesProcess(args.dataset_path, args.out_path, args.version)
    converter.convert()
