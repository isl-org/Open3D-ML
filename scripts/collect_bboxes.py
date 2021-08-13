import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from os import makedirs
import random
import argparse
import pickle

from tqdm import tqdm
from open3d.ml import datasets
from open3d.ml.datasets import utils


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
    parser.add_argument('--dataset_type',
                        help='Name of dataset class',
                        default="KITTI",
                        required=False)

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


if __name__ == '__main__':
    """Collect bboxes

    This script constructs a bbox dictionary for later data augmentation.

    Args:
        dataset_path (str): Directory to load dataset data.
        out_path (str): Directory to save pickle file (infos).
        dataset_type (str): Name of dataset object under `ml3d/datasets` to use to 
                            load the test data split from the dataset folder. Uses 
                            reflection to dynamically import this dataset object by name. 
                            Default: KITTI

    Example usage:

    python scripts/collect_bboxes.py --dataset_path /path/to/data --dataset_type MyCustomDataset
    """
    args = parse_args()
    out_path = args.out_path
    if out_path is None:
        out_path = args.dataset_path

    classname = getattr(datasets, args.dataset_type)
    dataset = classname(args.dataset_path)
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
