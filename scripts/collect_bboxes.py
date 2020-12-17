import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from os import makedirs
import random
import argparse
import pickle

from tqdm import tqdm
from ml3d.datasets import KITTI


def parse_args():
    parser = argparse.ArgumentParser(description='Collect bounding boxes for augmentation.')
    parser.add_argument('--dataset_path',
                        help='path to Dataset root',
                        required=True)
    parser.add_argument('--out_path',
                        help='Output path to store pickle',
                        required=True)

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


if __name__ == '__main__':
    args = parse_args()

    dataset = KITTI(args.dataset_path)
    train = dataset.get_split('train')

    bboxes = []
    for i in range(len(train)):
        bbox = train.get_data(i)['bounding_boxes']
        bboxes += bbox

    file = open(join(args.out_path, 'bboxes.pkl'), 'wb')
    pickle.dump(bboxes, file)
