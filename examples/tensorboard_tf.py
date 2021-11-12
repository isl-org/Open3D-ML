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
from os.path import isfile, join, dirname, abspath
import sys
import numpy as np
import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary
import tensorflow as tf

BASE_LOGDIR = "demo_logs/tf/"
DEMO_DATA_DIR = join(dirname(abspath(__file__)), "demo_data")
KITTI_LABELS = {
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


def semantic_segmentation():
    """Example writing 3D TensorBoard summary data for semantic segmentation"""
    label_files = tuple(
        join(DEMO_DATA_DIR, 'labels', fn)
        for fn in listdir(join(DEMO_DATA_DIR, 'labels'))
        if isfile(join(DEMO_DATA_DIR, 'labels', fn)))
    pcd_files = tuple(
        join(DEMO_DATA_DIR, 'points', fn)
        for fn in listdir(join(DEMO_DATA_DIR, 'points'))
        if isfile(join(DEMO_DATA_DIR, 'points', fn)))

    if len(pcd_files) == 0 or len(pcd_files) != len(label_files):
        print("No point cloud data or labels found.")
        sys.exit(1)

    rng = np.random.default_rng()

    logdir = join(BASE_LOGDIR, "semseg-example")
    writer = tf.summary.create_file_writer(logdir)
    for step in range(len(pcd_files)):
        # We will pretend these are the inputs and outputs of a Semantic
        # Segmentation model
        # float, shape (N, 3), or (B, N, 3) for a batch
        points = np.load(pcd_files[step])
        # int, shape (N, 1), or (B, N, 1) for a batch
        labels = np.load(label_files[step])
        # we can also visualize random scores
        random_scores = rng.random(labels.shape, dtype=np.float32)
        # and outputs of some pretend network layers. The first 3 dimensions
        # can be visualized as RGB colors
        random_features = np.hstack(
            (rng.random(labels.shape, dtype=np.float32),
             rng.random(labels.shape, dtype=np.float32) - 0.5,
             -rng.random(labels.shape, dtype=np.float32)))
        with writer.as_default():
            # You can use Torch tensors directly too.
            # Prefix the data with "vertex_" to indicate that this is per vertex
            # data.
            writer.add_3d("semantic_segmentation", {
                "vertex_positions": points,
                "vertex_labels": labels,
                "vertex_random_scores": random_scores,
                "vertex_random_features": random_features
            },
                          step,
                          label_to_names=KITTI_LABELS,
                          logdir=logdir)


def object_detection():
    """Example writing 3D TensorBoard summary data for object detection"""
    bbox_files = tuple(
        join(DEMO_DATA_DIR, 'bboxes', fn)
        for fn in listdir(join(DEMO_DATA_DIR, 'bboxes'))
        if isfile(join(DEMO_DATA_DIR, 'bboxes', fn)))
    pcd_files = tuple(
        join(DEMO_DATA_DIR, 'points', fn)
        for fn in listdir(join(DEMO_DATA_DIR, 'points'))
        if isfile(join(DEMO_DATA_DIR, 'points', fn)))

    if len(pcd_files) == 0 or len(pcd_files) != len(bbox_files):
        print("No point cloud data or bounding boxes found.")
        sys.exit(1)

    logdir = join(BASE_LOGDIR, "objdet-example")
    writer = tf.summary.create_file_writer(logdir)
    for step in range(len(pcd_files)):
        # We will pretend these are the inputs and outputs of an Object
        # Detection model
        # float, shape (N, 3), or (B, N, 3) for a batch
        points = np.load(pcd_files[step])
        # ????
        bboxes = np.load(bbox_files[step])
        with writer.as_default():
            # You can use Torch tensors directly too.
            writer.add_3d("input_pointcloud", {"vertex_positions": points},
                          step,
                          logdir=logdir)
            # Write bounding boxes in a separate call
            writer.add_3d("object_detection", {"bboxes": bboxes},
                          step,
                          label_to_names=KITTI_LABELS,
                          logdir=logdir)


if __name__ == "__main__":
    print("Writing example summary for semantic segmentation...")
    semantic_segmentation()
    print("Writing example summary for object detection...")
    object_detection()
