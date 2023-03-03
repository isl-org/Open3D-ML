#!/usr/bin/env python
import logging
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import numpy as np
import os
import sys
from os.path import exists, join, dirname

from util import ensure_demo_data

example_dir = os.path.dirname(os.path.realpath(__file__))

log = logging.getLogger(__name__)


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
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label_r[0] = 0

        results_k = pipeline_k.run_inference(data)
        pred_label_k = (results_k['predict_labels'] + 1).astype(np.int32)
        # Fill "unlabeled" value because predictions have no 0 values.
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


def get_torch_ckpts():
    kpconv_url = "https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth"
    randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth"

    ckpt_path_r = example_dir + "/vis_weights_{}.pth".format('RandLANet')
    if not exists(ckpt_path_r):
        cmd = "wget {} -O {}".format(randlanet_url, ckpt_path_r)
        os.system(cmd)

    ckpt_path_k = example_dir + "/vis_weights_{}.pth".format('KPFCNN')
    if not exists(ckpt_path_k):
        cmd = "wget {} -O {}".format(kpconv_url, ckpt_path_k)
        print(cmd)
        os.system(cmd)

    return ckpt_path_r, ckpt_path_k


def get_tf_ckpts():
    kpconv_url = "https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202010021102utc.zip"
    randlanet_url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.zip"

    ckpt_path_dir = example_dir + "/vis_weights_{}".format('RandLANet')
    if not exists(ckpt_path_dir):
        ckpt_path_zip = example_dir + "/vis_weights_{}.zip".format('RandLANet')
        cmd = "wget {} -O {}".format(randlanet_url, ckpt_path_zip)
        os.system(cmd)
        cmd = "unzip -j -o {} -d {}".format(ckpt_path_zip, ckpt_path_dir)
        os.system(cmd)
    ckpt_path_r = example_dir + "/vis_weights_{}/{}_{}".format(
        'RandLANet', 'randlanet', 'semantickitti')

    ckpt_path_dir = example_dir + "/vis_weights_{}".format('KPFCNN')
    if not exists(ckpt_path_dir):
        ckpt_path_zip = example_dir + "/vis_weights_{}.zip".format('KPFCNN')
        cmd = "wget {} -O {}".format(kpconv_url, ckpt_path_zip)
        os.system(cmd)
        cmd = "unzip -j -o {} -d {}".format(ckpt_path_zip, ckpt_path_dir)
        os.system(cmd)
    ckpt_path_k = example_dir + "/vis_weights_{}/{}".format('KPFCNN', 'ckpt-1')

    return ckpt_path_r, ckpt_path_k


# ------------------------------


def main():
    kitti_labels = ml3d.datasets.SemanticKITTI.get_label_to_names()
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(kitti_labels.keys()):
        lut.add_label(kitti_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    # load pretrained weights depending on used ml framework (torch or tf)
    if ("open3d.ml.torch" in sys.modules):  # torch is used
        ckpt_path_r, ckpt_path_k = get_torch_ckpts()
    else:  # tf is used
        ckpt_path_r, ckpt_path_k = get_tf_ckpts()

    model = ml3d.models.RandLANet(ckpt_path=ckpt_path_r)
    pipeline_r = ml3d.pipelines.SemanticSegmentation(model)
    pipeline_r.load_ckpt(model.cfg.ckpt_path)

    model = ml3d.models.KPFCNN(ckpt_path=ckpt_path_k)
    pipeline_k = ml3d.pipelines.SemanticSegmentation(model)
    pipeline_k.load_ckpt(model.cfg.ckpt_path)

    data_path = ensure_demo_data()
    pc_names = ["000700", "000750"]
    pcs = get_custom_data(pc_names, data_path)
    pcs_with_pred = pred_custom_data(pc_names, pcs, pipeline_r, pipeline_k)

    v.visualize(pcs_with_pred)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    main()
