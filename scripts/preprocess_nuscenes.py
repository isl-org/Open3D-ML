try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
except ImportError:
    raise ImportError('Please run "pip install nuscenes-devkit" '
                      'to install the official devkit first.')

import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from os import makedirs
import random
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess NuScenes Dataset.')
    parser.add_argument('--dataset_path',
                        help='path to Nuscene root',
                        required=True)
    parser.add_argument('--out_path',
                        help='Output path to store infos',
                        required=True)

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

            assert os.path.exists(lidar_path)

            data = {
                'lidar_path': lidar_path,
                'token': sample['token'],
                'lidar2ego_tr': calib_rec['translation'],
                'lidar2ego_rot': calib_rec['rotation'],
                'ego2global_tr': pose_rec['translation'],
                'ego2global_rot': pose_rec['rotation'],
                'timestamp': sample['timestamp']
            }

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
    args = parse_args()
    converter = NuScenesProcess(args.dataset_path, args.out_path, args.version)
    converter.convert()
