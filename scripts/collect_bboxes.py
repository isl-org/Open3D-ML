import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from os import makedirs
import random
import argparse
import pickle

from tqdm import tqdm
from open3d.ml.datasets import KITTI, utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect bounding boxes for augmentation.')
    parser.add_argument('--dataset_path',
                        help='path to Dataset root',
                        required=True)
    parser.add_argument(
        '--out_path',
        help='Output path to store pickle (default to dataet_path)',
        default=None,
        required=False)

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


if __name__ == '__main__':
    args = parse_args()
    out_path = args.out_path
    if out_path is None:
        out_path = args.dataset_path

    dataset = KITTI(args.dataset_path)
    train = dataset.get_split('train')

    bboxes = []
    for i in tqdm(range(len(train))):
        data = train.get_data(i)
        bbox = data['bounding_boxes']
        flat_bbox = [box.to_xyzwhlr() for box in bbox]

        indices = utils.operations.points_in_box(data['point'], flat_bbox)
        for i, box in enumerate(bbox):
            pts = data['point'][indices[:, i]]
            box.points_inside_box = pts
            bboxes.append(box)

    file = open(join(out_path, 'bboxes.pkl'), 'wb')
    pickle.dump(bboxes, file)
