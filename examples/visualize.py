#!/usr/bin/env python
import open3d.ml.torch as ml3d
from open3d.ml.datasets import (SemanticKITTI, ParisLille3D, Semantic3D, S3DIS,
                                Toronto3D)
from open3d.ml.vis import Visualizer, LabelLUT
from open3d.ml.utils import get_module

import argparse
import math
import numpy as np
import os
import random
import sys
import tensorflow as tf
import torch
from os.path import exists, join, isfile, dirname, abspath, split


def print_usage_and_exit():
    print(
        "Usage: ml-test.py [kitti|paris|toronto|semantic3d|s3dis|custom] path/to/dataset"
    )
    exit(0)


# ------ for custom data -------
kitti_labels = {
    0: 'unlabeled',
    1: 'car',
    2: 'bicycle',
    3: 'motorcycle',
    4: 'truck',
    5: 'other-vehicle',
    6: 'person',
    7: 'bicyclist',
    8: 'motorcyclist',
    9: 'road',
    10: 'parking',
    11: 'sidewalk',
    12: 'other-ground',
    13: 'building',
    14: 'fence',
    15: 'vegetation',
    16: 'trunk',
    17: 'terrain',
    18: 'pole',
    19: 'traffic-sign'
}


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Datasets')
    parser.add_argument('dataset_name')
    parser.add_argument('dataset_path')
    parser.add_argument('--model', default='RandLANet')

    args = parser.parse_args()

    return args


def get_custom_data(pc_names, path):

    pc_data = []
    for i, name in enumerate(pc_names):
        pc_path = join(path, 'points', name + '.npy')
        label_path = join(path, 'labels', name + '.npy')
        point = np.load(pc_path)[:, 0:3]
        label = np.squeeze(np.load(label_path))

        data = {
            'point': point,
            'feat': None,
            'label': label,
        }
        pc_data.append(data)

    return pc_data


def pred_custom_data(pc_names, pcs, pipeline_r, pipeline_k):
    vis_points = []
    for i, data in enumerate(pcs):
        name = pc_names[i]

        results_r = pipeline_r.run_inference(data)
        pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
        pred_label_r[0] = 0

        results_k = pipeline_k.run_inference(data)
        pred_label_k = (results_k['predict_labels'] + 1).astype(np.int32)
        pred_label_k[0] = 0

        label = data['label']
        pts = data['point']

        vis_d = {
            "name": name,
            "points": pts,
            "labels": label,
            "pred": pred_label_k,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_randlanet",
            "points": pts,
            "labels": pred_label_r,
        }
        vis_points.append(vis_d)

        vis_d = {
            "name": name + "_kpconv",
            "points": pts,
            "labels": pred_label_k,
        }
        vis_points.append(vis_d)

    return vis_points


# ------------------------------

from open3d.ml.torch.pipelines import SemanticSegmentation
from open3d.ml.torch.models import RandLANet, KPFCNN


def main():
    args = parse_args()

    which = args.dataset_name
    path = args.dataset_path

    if which == "kitti":
        dataset = SemanticKITTI(path)
    elif which == "paris":
        dataset = ParisLille3D(path)
    elif which == "toronto":
        dataset = Toronto3D(path)
    elif which == "semantic3d":
        dataset = Semantic3D(path)
    elif which == "s3dis":
        dataset = S3DIS(path)
    elif which == "custom":
        dataset = None
    else:
        print("[ERROR] '" + which + "' is not a valid dataset")
        print_usage_and_exit()

    v = Visualizer()
    if dataset is None:  # custom
        lut = LabelLUT()
        for val in sorted(kitti_labels.keys()):
            lut.add_label(kitti_labels[val], val)
        v.set_lut("labels", lut)
        v.set_lut("pred", lut)
        path = os.path.dirname(os.path.realpath(__file__)) + "/demo_data"

        kpconv_url = "https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth"
        randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202009090354utc.pth"
        ckpt_path = "../dataset/checkpoints/vis_weights_{}.pth".format(
            args.model)

        pc_names = ["000700", "000750"]

        ckpt_path = "../dataset/checkpoints/vis_weights_{}.pth".format(
            'RandLANet')
        if not exists(ckpt_path):
            cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
            os.system(cmd)
        model = RandLANet(ckpt_path=ckpt_path)
        pipeline_r = SemanticSegmentation(model)
        pipeline_r.load_ckpt(model.cfg.ckpt_path)

        ckpt_path = "../dataset/checkpoints/vis_weights_{}.pth".format('KPFCNN')
        if not exists(ckpt_path):
            cmd = "wget {} -O {}".format(kpconv_url, ckpt_path)
            os.system(cmd)
        model = KPFCNN(ckpt_path=ckpt_path, in_radius=10)
        pipeline_k = SemanticSegmentation(model)
        pipeline_k.load_ckpt(model.cfg.ckpt_path)

        pcs = get_custom_data(pc_names, path)
        pcs_with_pred = pred_custom_data(pc_names, pcs, pipeline_r, pipeline_k)

        v.visualize(pcs_with_pred)
    else:
        v.visualize_dataset(dataset, "training")


if __name__ == "__main__":
    main()
