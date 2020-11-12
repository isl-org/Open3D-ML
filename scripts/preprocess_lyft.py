try:
    from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
except ImportError:
    raise ImportError('Please run "pip install lyft_dataset_sdk" '
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
    parser = argparse.ArgumentParser(description='Preprocess Lyft Dataset.')
    parser.add_argument('--dataset_path',
                        help='path to Lyft root',
                        required=True)
    parser.add_argument('--out_path',
                        help='Output path to store infos',
                        required=True)

    parser.add_argument('--version',
                        help='one of {v1.01-train, v1.01-test, sample}',
                        default='v1.01-train')

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


class LyftProcess():
    """Preprocess Lyft.
    This class collects paths and labels using lyft-devkit.
    Args:
        dataset_path (str): Directory to load lyft data.
        out_path (str): Directory to save pickle file(infos).
        version (str): version of dataset. Default: v1.01-train.
    """

    def __init__(self, dataset_path, out_path, version='v1.01-train'):

        assert version in ['v1.01-train', 'v1.01-test', 'sample']
        self.is_test = 'test' in version
        self.out_path = out_path

        self.lyft = Lyft(data_path=join(dataset_path, version),
                         json_path=join(dataset_path, version, 'data'),
                         verbose=True)

        if version == 'v1.01-train':
            train_scenes = open(
                join(dirname(__file__),
                     '../ml3d/datasets/_resources/lyft/train.txt'),
                'r').read().split('\n')
            val_scenes = open(
                join(dirname(__file__),
                     '../ml3d/datasets/_resources/lyft/val.txt'),
                'r').read().split('\n')
        elif version == 'v1.0-test':
            train_scenes = open(
                join(dirname(__file__),
                     '../ml3d/datasets/_resources/lyft/test.txt'),
                'r').read().split('\n')
            val_scenes = []
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
            'bicycle': 'bicycle',
            'bus': 'bus',
            'car': 'car',
            'emergency_vehicle': 'emergency_vehicle',
            'motorcycle': 'motorcycle',
            'other_vehicle': 'other_vehicle',
            'pedestrian': 'pedestrian',
            'truck': 'truck',
            'animal': 'animal'
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
        lyft = self.lyft
        train_info = []
        val_info = []

        for sample in tqdm(lyft.sample):
            lidar_token = sample['data']['LIDAR_TOP']
            sd_rec = lyft.get('sample_data', lidar_token)
            calib_rec = lyft.get('calibrated_sensor',
                                 sd_rec['calibrated_sensor_token'])
            pose_rec = lyft.get('ego_pose', sd_rec['ego_pose_token'])

            lidar_path, boxes, _ = lyft.get_sample_data(lidar_token)

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
                    lyft.get('sample_annotation', token)
                    for token in sample['anns']
                ]
                locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                rots = np.array([
                    b.orientation.yaw_pitch_roll[0] for b in boxes
                ]).reshape(-1, 1)

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

            if sample['scene_token'] in self.train_scenes:
                train_info.append(data)
            else:
                val_info.append(data)

        return train_info, val_info

    def get_available_scenes(self):
        lyft = self.lyft
        available_scenes = []

        for scene in lyft.scene:
            token = scene['token']
            scene_rec = lyft.get('scene', token)
            sample_rec = lyft.get('sample', scene_rec['first_sample_token'])
            sample_data = lyft.get('sample_data',
                                   sample_rec['data']['LIDAR_TOP'])

            lidar_path, boxes, _ = lyft.get_sample_data(sample_data['token'])
            lidar_path = str(lidar_path)

            if not os.path.exists(lidar_path):
                continue
            else:
                available_scenes.append(scene)

        return available_scenes


if __name__ == '__main__':
    args = parse_args()
    converter = LyftProcess(args.dataset_path, args.out_path, args.version)
    converter.convert()
