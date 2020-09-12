#!/usr/bin/env python
from ml3d.datasets import Semantic3D
from ml3d.datasets import SemanticKITTI
from ml3d.datasets import ParisLille3D
from ml3d.datasets import Toronto3D
from ml3d.vis import Visualizer
import random
import sys

def print_usage_and_exit():
    print("Usage: ml-test.py [kitti|paris|toronto|sematic3d] path/to/dataset")
    exit(0)

def create_custom_dataset():
    d1 = { "name": "random1",
           "points": create_uniform_xyz(200),
           "feature": create_random_feature(200),
           "random": create_random_feature(200),
         }
    d2 = { "name": "random2",
           "points": create_uniform_xyz(200),
           "feature": create_random_feature(200),
         }
    d3 = { "name": "random3-no-data",
           "points": create_uniform_xyz(200),
         }
    return [d1, d2, d3]

def create_uniform_xyz(n_points):
    return [[random.random(), random.random(), 0.0] for i in range(0, n_points)]

def create_random_feature(n_points):
    return [random.random() for i in range(0, n_points)]

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
        v.visualize(create_custom_dataset())
    else:
        #v.visualize_dataset(dataset, "training")  # everything
        v.visualize_dataset(dataset, "training", [0])
        #v.visualize_dataset(dataset, "training", [0, 2, 4, 6])
        #v.visualize_dataset(dataset, "training", range(0, 4))
        #v.visualize_dataset(dataset, "training", range(0, 20))

if __name__ == "__main__":
    main()
