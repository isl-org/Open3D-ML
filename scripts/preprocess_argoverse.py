try:
    from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
except ImportError:
    raise ImportError('Please clone and install agroverse-api.')

import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from os import makedirs
import random
import argparse
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess Argoverse Dataset.')
    parser.add_argument('--dataset_path',
                        help='path to Argoverse root',
                        required=True)
    parser.add_argument('--out_path',
                        help='Output path to store infos',
                        required=True)

    parser.add_argument('--version',
                        help='one of {train, val, test, sample}',
                        default='sample')

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


class ArgoverseProcess():
    """Preprocess Argoverse.
    This class collects paths and labels using argoverse-api.
    Args:
        dataset_path (str): Directory to load argoverse data.
        out_path (str): Directory to save pickle file(infos).
        version (str): version of dataset. Default: sample.
    """

    def __init__(self, dataset_path, out_path, version='sample'):

        assert version in ['train', 'val', 'test', 'sample']

        self.is_test = 'test' in version
        self.out_path = out_path

        self.dataset_path = join(dataset_path, version)

        self.argo = ArgoverseTrackingLoader(self.dataset_path)

        print(f"Total number of logs : {len(self.argo)}, version : {version}")

        self.version = version

    def convert(self):
        info = []
        for scene in self.argo:
            info.append(self.process_scene(scene))

        out_path = self.out_path
        makedirs(out_path, exist_ok=True)

        with open(join(out_path, f'infos_{self.version}.pkl'), 'wb') as f:
            pickle.dump(info, f)
            print(
                f"Saved {self.version} info at {join(out_path, f'infos_{self.version}.pkl')}"
            )

    def process_scene(self, scene):
        info = {}

        num_pc = scene.lidar_count
        lidar_path = scene.lidar_list

        info['num_pc'] = num_pc
        info['lidar_path'] = lidar_path

        if self.is_test:
            info['bbox'] = []
            return info

        bbox_all = []

        for idx in tqdm(range(num_pc)):
            boxes = []
            labels = scene.get_label_object(idx)

            for label in labels:
                box = {}
                box['l'] = label.length
                box['w'] = label.width
                box['h'] = label.height
                box['3d_coord'] = label.as_3d_bbox()
                box['2d_coord'] = label.as_2d_bbox()
                box['label_class'] = label.label_class
                box['occlusion'] = label.occlusion
                box['center'] = label.translation
                box['quaternion'] = label.quaternion

                boxes.append(box)

            bbox_all.append(boxes)

        info['bbox'] = bbox_all

        return info


if __name__ == '__main__':
    args = parse_args()
    converter = ArgoverseProcess(args.dataset_path, args.out_path, args.version)
    converter.convert()
