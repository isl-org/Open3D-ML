import torch
import numpy as np
import open3d.ml as ml3d
import math

from ml3d.vis import Visualizer, BoundingBox3D, LabelLUT
from ml3d.datasets import KITTI

from ml3d.torch.pipelines import ObjectDetection
from ml3d.torch.models import PointPillars
from ml3d.torch.dataloaders import TorchDataloader

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for inference of object detection')
    parser.add_argument('--path_kitti', help='path to KITTI', required=True)
    parser.add_argument('--path_ckpt_pointpillars',
                        help='path to PointPillars checkpoint')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def main(args):

    model = PointPillars(voxel_size=[0.16, 0.16, 4],
                         point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                         loss={
                            "focal": {
                                "gamma": 2.0,
                                "alpha": 0.25,
                                "loss_weight": 1.0
                            },
                            "smooth_l1": {
                                "beta": 0.11,
                                "loss_weight": 2.0
                            },
                            "cross_entropy": {
                                "loss_weight": 0.2
                            }
                         })
    dataset = KITTI(args.path_kitti)
    pipeline = ObjectDetection(model, dataset, device="gpu")

    # load the parameters.
    pipeline.load_ckpt(ckpt_path='/Users/lprantl/Open3D-ML/checkpoints/pointpillars_kitt_3class_mmdet.pth')

    test_split = TorchDataloader(dataset=dataset.get_split('training'),
                                 preprocess=model.preprocess,
                                 transform=model.transform,
                                 use_cache=False,
                                 shuffle=False)
    data = test_split[5]['data']

    # run inference on a single example.
    result = pipeline.run_inference(data)

    boxes = data['bounding_boxes']
    boxes.extend(result)

    vis = Visualizer()

    lut = LabelLUT()
    for val in sorted(dataset.label_to_names.keys()):
        lut.add_label(dataset.label_to_names[val], val)

    vis.visualize([{
        "name": "KITTI",
        'points': data['point'][:, :3]
    }],
                  lut,
                  bounding_boxes=boxes)


if __name__ == '__main__':
    args = parse_args()
    main(args)
