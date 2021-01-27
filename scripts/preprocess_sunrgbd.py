import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from os import makedirs
import random
import argparse
import json
import csv
import scipy.io
import imageio
import pickle

from shutil import copyfile
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess SunRGBD Dataset.')
    parser.add_argument('--dataset_path',
                        help='path to Scannet scans directory',
                        required=True)
    parser.add_argument('--out_path',
                        help='Output path to store pickle data.',
                        default=None,
                        required=False)

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


class SunRGBDProcess():
    """Preprocess SunRGBD.
    This class converts Sunrgbd raw data into npy files.
    Args:
        dataset_path (str): Directory to load sunrgbd data.
        out_path (str): Directory to save pickle file(infos).
    """

    def __init__(self, dataset_path, out_path):

        self.out_path = out_path
        self.dataset_path = dataset_path

        allsplit = scipy.io.loadmat(
            join(dataset_path, 'SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'))
        train_split = allsplit['alltrain'][0]
        val_split = allsplit['alltest'][0]

        train_paths = []
        val_paths = []
        for i in range(train_split.shape[0]):
            path = train_split[i][0][17:]
            if path[-1] == '/':
                path = path[:-1]
            train_paths.append(path)
        for i in range(val_split.shape[0]):
            path = val_split[i][0][17:]
            if path[-1] == '/':
                path = path[:-1]
            val_paths.append(path)

        print(
            f"Total scans : train : {len(train_paths)}, val : {len(val_paths)}")

        self.train_idx = []
        self.val_idx = []

        self.meta3 = scipy.io.loadmat(
            join(dataset_path, 'SUNRGBDMeta3DBB_v2.mat'))['SUNRGBDMeta'][0]
        self.meta2 = scipy.io.loadmat(
            join(dataset_path, 'SUNRGBDMeta2DBB_v2.mat'))['SUNRGBDMeta2DBB'][0]
        for i in range(self.meta3.shape[0]):
            path = self.meta3[i][0][0]
            if path in train_paths:
                self.train_idx.append(i)
            elif path in val_paths:
                self.val_idx.append(i)
            else:
                raise ValueError(f"{path} not found")

        self.create_dirs()

        with open(join(self.out_path, 'sunrgbd_trainval/train_data_idx.txt'),
                  'w') as f:
            for idx in self.train_idx:
                f.write(str(idx) + '\n')

        with open(join(self.out_path, 'sunrgbd_trainval/val_data_idx.txt'),
                  'w') as f:
            for idx in self.val_idx:
                f.write(str(idx) + '\n')

    def create_dirs(self):
        os.makedirs(join(self.out_path, 'sunrgbd_trainval', 'depth'),
                    exist_ok=True)
        os.makedirs(join(self.out_path, 'sunrgbd_trainval', 'image'),
                    exist_ok=True)
        os.makedirs(join(self.out_path, 'sunrgbd_trainval', 'calib'),
                    exist_ok=True)
        os.makedirs(join(self.out_path, 'sunrgbd_trainval', 'label'),
                    exist_ok=True)

    def convert(self):
        for imageid in tqdm(range(0, 10335)):
            self.process_scene(imageid)

    def process_scene(self, imageid):
        meta2 = self.meta2[imageid]
        meta3 = self.meta3[imageid]
        meta3 = self.data2dict(meta3)

        # Save points_rgb
        points_rgb = self.read3dpoints(meta3)
        np.save(
            join(self.out_path, 'sunrgbd_trainval', 'depth', f'{imageid}.npy'),
            points_rgb)

        # Save Image
        copyfile(
            join(self.dataset_path, str(meta3['rgbpath'][17:])),
            join(self.out_path, 'sunrgbd_trainval/image', f'{imageid}.jpg'))

        # Save label
        labels = []
        save_2d_box = True
        for i in range(len(meta3['3DBB'])):
            box = meta3['3DBB'][i]
            try:
                box2d = meta2[1][0][i]
            except:
                save_2d_box = False

            if save_2d_box and box2d[2][0] not in box['classname']:
                save_2d_box = False

            if save_2d_box:
                box2d = box2d[1][0]
                label = [
                    box['classname'], box['centroid'][0], box['centroid'][1],
                    box['centroid'][2], box['coeff'][0], box['coeff'][1],
                    box['coeff'][2], box['orientation'][0],
                    box['orientation'][1], box2d[0], box2d[1], box2d[2],
                    box2d[3]
                ]
            else:
                label = [
                    box['classname'], box['centroid'][0], box['centroid'][1],
                    box['centroid'][2], box['coeff'][0], box['coeff'][1],
                    box['coeff'][2], box['orientation'][0],
                    box['orientation'][1]
                ]

            labels.append(label)

        with open(
                join(self.out_path, 'sunrgbd_trainval/label', f'{imageid}.pkl'),
                'wb') as f:
            pickle.dump(labels, f)

    def data2dict(self, data):
        dat = {}
        dat['seqname'] = data[0][0]
        dat['Rtilt'] = data[1]
        dat['K'] = data[2]
        dat['depthpath'] = data[3][0]
        dat['rgbpath'] = data[4][0]
        dat['anno_extrinsics'] = data[5]
        dat['depthname'] = data[6][0]
        dat['rgbname'] = data[7][0]
        dat['sensor'] = data[8][0]
        dat['valid'] = data[9][0][0]

        labels = []
        for i in range(data[10][0].shape[0]):
            label = {}
            label['basis'] = data[10][0][i][0]
            label['coeff'] = data[10][0][i][1][0]
            label['centroid'] = data[10][0][i][2][0]
            label['classname'] = data[10][0][i][3][0]
            label['seqname'] = data[10][0][i][4][0]
            label['orientation'] = data[10][0][i][5][0]
            label['label'] = data[10][0][i][6][0][0]
            labels.append(label)

        dat['3DBB'] = labels

        return dat

    def read3dpoints(self, data):
        depth_path = join(self.dataset_path, str(data['depthpath'][17:]))
        depth = imageio.imread(depth_path)
        depth = (depth >> 3) | (depth << 13)
        depth = np.array(depth, np.float32) / 1000

        cx = data['K'][0][2]
        cy = data['K'][1][2]
        fx = data['K'][0][0]
        fy = data['K'][1][1]

        if data['rgbpath'] != '':
            img = imageio.imread(
                join(self.dataset_path, str(data['rgbpath'][17:])))
            img = np.array(img, np.float32) / 255
        else:
            img = np.array((depth.shape[0], depth.shape[1], 3), np.float32)
            img[:, :, 1] = 1

        invalid = depth == 0

        x, y = np.meshgrid([i for i in range(1, depth.shape[1] + 1)],
                           [j for j in range(1, depth.shape[0] + 1)])
        x3 = (x - cx) * depth * 1.0 / fx
        y3 = (y - cy) * depth * 1.0 / fy
        z3 = depth

        points = np.concatenate(
            [x3.reshape(-1, 1),
             z3.reshape(-1, 1), -y3.reshape(-1, 1)], axis=1)

        points = np.transpose(np.matmul(data['Rtilt'], np.transpose(points)))

        img = img.reshape(-1, 3)
        img = img[points[:, 1] != 0]
        points = points[points[:, 1] != 0]
        points_img = np.concatenate([points, img], axis=1)

        return points_img


if __name__ == '__main__':
    args = parse_args()
    out_path = args.out_path
    if out_path is None:
        args.out_path = args.dataset_path
    converter = SunRGBDProcess(args.dataset_path, args.out_path)
    converter.convert()
