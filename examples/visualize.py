#!/usr/bin/env python
from ml3d.datasets import Semantic3D
from ml3d.datasets import SemanticKITTI
from ml3d.datasets import ParisLille3D
from ml3d.datasets import Toronto3D
from ml3d.vis import Visualizer
import sys

def print_usage_and_exit():
    print("Usage: ml-test.py [kitti|paris|toronto|sematic3d] path/to/dataset")
    exit(0)

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
    else:
        print("[ERROR] '" + which + "' is not a valid dataset")
        print_usage_and_exit()

    data = dataset.get_split("training")
    if len(data) == 0:
        print("[WARNING] no data!")
        exit(0)

    # Training data is randomized. Sort, so that the same index always returns
    # the same piece of data.
    path2idx = {}
    for i in range(0, len(data.path_list)):
        path2idx[data.path_list[i]] = i
    indices = [path2idx[p] for p in sorted(path2idx.keys())]

    # Visualize
    Visualizer.visualize(data, [indices[0]])
    #Visualizer.visualize(data, [indices[0], indices[2], indices[4], indices[6]])
    #Visualizer.visualize(data, indices)
    #Visualizer.visualize(data, indices[:4])
    #Visualizer.visualize(data, indices[:20])

if __name__ == "__main__":
    main()
