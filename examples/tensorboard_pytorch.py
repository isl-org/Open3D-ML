# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import open3d as o3d
import open3d.ml.torch as ml3d
# pylint: disable-next=unused-import
from open3d.visualization.tensorboard_plugin import summary
from torch.utils.tensorboard import SummaryWriter

from util import ensure_demo_data

BASE_LOGDIR = "demo_logs/pytorch/"


def semantic_segmentation(DEMO_DATA_DIR):
    """Example writing 3D TensorBoard summary data for semantic segmentation"""
    SEMANTIC_KITTI_LABELS = {
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
    labels_dir = join(DEMO_DATA_DIR, 'SemanticKITTI', 'labels')
    label_files = tuple(
        join(labels_dir, fn)
        for fn in listdir(labels_dir)
        if isfile(join(labels_dir, fn)))
    points_dir = join(DEMO_DATA_DIR, 'SemanticKITTI', 'points')
    pcd_files = tuple(
        join(points_dir, fn)
        for fn in listdir(points_dir)
        if isfile(join(points_dir, fn)))

    if len(pcd_files) == 0 or len(pcd_files) != len(label_files):
        print("No point cloud data or labels found.")
        sys.exit(1)

    rng = np.random.default_rng()

    writer = SummaryWriter(join(BASE_LOGDIR, "semseg-example"))
    for step in range(len(pcd_files)):
        # We will pretend these are the inputs and outputs of a Semantic
        # Segmentation model
        # float, shape (N, 3), or (B, N, 3) for a batch
        points = np.load(pcd_files[step])
        # int, shape (N, 1), or (B, N, 1) for a batch
        labels = np.load(label_files[step])
        # We can also visualize noisy scores (car, road, vegetation)
        scores = np.hstack((labels == 1, labels == 9, labels == 15))
        scores = np.clip(scores + rng.normal(0., 0.05, size=scores.shape), 0.,
                         1.)
        # and outputs of some pretend network layers. The first 3 dimensions
        # can be visualized as RGB colors. Here we will use distances from the
        # centroids of (all points, road, vegetation).
        centroid_all = np.mean(points, axis=0)
        d_all = np.linalg.norm(points - centroid_all, axis=1)
        centroid_road = np.mean(points[np.squeeze(labels) == 9, :], axis=0)
        d_road = np.linalg.norm(points - centroid_road, axis=1)
        centroid_vegetation = np.mean(points[np.squeeze(labels) == 15, :],
                                      axis=0)
        d_vegetation = np.linalg.norm(points - centroid_vegetation, axis=1)
        features = np.stack((d_all, d_road, d_vegetation), axis=1)

        # You can use Torch tensors directly too.
        # Prefix the data with "vertex_" for per vertex data.
        writer.add_3d(
            "semantic_segmentation",
            {
                "vertex_positions": points,  # (N, 3)
                "vertex_labels": labels,  # (N, 1)
                "vertex_scores": scores,  # (N, 3)
                "vertex_features": features  # (N, 3)
            },
            step,
            label_to_names=SEMANTIC_KITTI_LABELS)


def object_detection(DEMO_DATA_DIR):
    """Example writing 3D TensorBoard summary data for object detection"""
    dset = ml3d.datasets.KITTI(dataset_path=join(DEMO_DATA_DIR, 'KITTI'))
    val_split = dset.get_split('validation')
    name_to_labels = {
        name: label for label, name in dset.get_label_to_names().items()
    }
    if len(val_split) == 0:
        print("No point cloud data or bounding boxes found.")
        sys.exit(1)

    writer = SummaryWriter(join(BASE_LOGDIR, "objdet-example"))
    for step in range(len(val_split)):  # one pointcloud per step
        data = val_split.get_data(step)
        # We will pretend these are the inputs and outputs of an Object
        # Detection model. You can use Torch tensors directly too.
        writer.add_3d(
            "input_pointcloud",
            {  # float, shape (N, 3), or (B, N, 3) for a batch
                "vertex_positions": data['point'][:, :3],
                # Extra features: float, shape (N, 1), or (B, N, 1) for a batch
                # [should not be (N,)]
                "vertex_intensities": data['point'][:, 3:]
            },
            step)
        # We need label_class to be int, not str
        for bb in data['bounding_boxes']:
            if not isinstance(bb.label_class, int):
                bb.label_class = name_to_labels[bb.label_class]
        # Bounding boxes (pretend model output): (Nbb, ) or (B, Nbb) for a batch
        # Write bounding boxes in a separate call.
        writer.add_3d("object_detection", {"bboxes": data['bounding_boxes']},
                      step,
                      label_to_names=dset.get_label_to_names())


if __name__ == "__main__":
    DEMO_DATA_DIR = ensure_demo_data()
    print("Writing example summary for semantic segmentation...")
    semantic_segmentation(DEMO_DATA_DIR)
    print("Writing example summary for object detection...")
    object_detection(DEMO_DATA_DIR)
