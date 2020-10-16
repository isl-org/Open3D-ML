#!/usr/bin/env python
import open3d.ml.torch as ml3d
import argparse
import math
import numpy as np
import os
import random
import sys
import torch
from os.path import exists, join, isfile, dirname, abspath, split

example_dir = os.path.dirname(os.path.realpath(__file__))


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


def main():
    kitti_labels = ml3d.datasets.SemanticKITTI.get_label_to_names()
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(kitti_labels.keys()):
        lut.add_label(kitti_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    kpconv_url = "https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth"
    randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202009090354utc.pth"

    ckpt_path = example_dir + "/vis_weights_{}.pth".format('RandLANet')
    if not exists(ckpt_path):
        cmd = "wget {} -O {}".format(randlanet_url, ckpt_path)
        os.system(cmd)
    model = ml3d.models.RandLANet(ckpt_path=ckpt_path)
    pipeline_r = ml3d.pipelines.SemanticSegmentation(model)
    pipeline_r.load_ckpt(model.cfg.ckpt_path)

    ckpt_path = example_dir + "/vis_weights_{}.pth".format('KPFCNN')
    if not exists(ckpt_path):
        cmd = "wget {} -O {}".format(kpconv_url, ckpt_path)
        print(cmd)
        os.system(cmd)
    model = ml3d.models.KPFCNN(ckpt_path=ckpt_path, in_radius=10)
    pipeline_k = ml3d.pipelines.SemanticSegmentation(model)
    pipeline_k.load_ckpt(model.cfg.ckpt_path)

    data_path = example_dir + "/demo_data"
    pc_names = ["000700", "000750"]
    pcs = get_custom_data(pc_names, data_path)
    pcs_with_pred = pred_custom_data(pc_names, pcs, pipeline_r, pipeline_k)

    v.visualize(pcs_with_pred)


if __name__ == "__main__":
    main()
