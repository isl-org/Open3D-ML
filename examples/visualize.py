#!/usr/bin/env python
from ml3d.datasets import Semantic3D
from ml3d.datasets import SemanticKITTI
from ml3d.datasets import ParisLille3D
from ml3d.datasets import Toronto3D
from ml3d.vis import Visualizer, LabelLUT
import math
import numpy as np
import os
import random
import sys

def print_usage_and_exit():
    print("Usage: ml-test.py [kitti|paris|toronto|sematic3d|custom] path/to/dataset")
    exit(0)

# ------ for custom data -------
kitti_labels = { 0: 'unlabeled',
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
  
def create_custom_dataset(path):
    pts = np.load(path + "/kitti_01_points_000000.npy")
    d1 = { "name": "kitti01 000000 + dist + random",
           "points": pts,
           "labels": np.load(path + "/kitti_01_labels_000000.npy"),
           "distance": create_distance(pts),
           "random": create_random_feature(len(pts)),
         }
    pts = np.load(path + "/kitti_01_points_000001.npy")
    d2 = { "name": "kitti01 000001 + dist",
           "points": pts,
           "labels": np.load(path + "/kitti_01_labels_000001.npy"),
           "distance": create_distance(pts)
         }
    d3 = { "name": "kitti01 000002",
           "points": np.load(path + "/kitti_01_points_000002.npy"),
           "labels": np.load(path + "/kitti_01_labels_000002.npy")
         }
    return [d1, d2, d3]

def create_distance(pts):
    return [math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]) for p in pts]

def create_random_feature(n_points):
    return [random.random() for i in range(0, n_points)]
# ------------------------------

def main():
    if len(sys.argv) != 3:
        print_usage_and_exit()

    which = sys.argv[1]
    path = sys.argv[2]

    if which == "kitti":
        dataset = SemanticKITTI(path)
    elif which == "paris":
        dataset = ParisLille3D(path)
    elif which == "toronto":
        dataset = Toronto3D(path)
    elif which == "semantic3d":
        dataset = Semantic3D(path)
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

        path = os.path.dirname(os.path.realpath(__file__)) + "/data"
        v.visualize(create_custom_dataset(path))
    else:
        #v.visualize_dataset(dataset, "training")  # everything
        v.visualize_dataset(dataset, "training", [0])
        #v.visualize_dataset(dataset, "training", [0, 2, 4, 6])
        #v.visualize_dataset(dataset, "training", range(0, 4))
        #v.visualize_dataset(dataset, "training", range(0, 20))

if __name__ == "__main__":
    main()
