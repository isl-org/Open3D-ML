#!/usr/bin/env python
from ml3d.datasets import Semantic3D
from ml3d.datasets import SemanticKITTI
from ml3d.datasets import ParisLille3D
from ml3d.datasets import Toronto3D
from ml3d.vis import Visualizer
import sys

class KITTIConfig:
    def __init__(self, path):
        self.dataset_path = path + "/sequences"
        self.training_split = ['01']
        validation_split = ['08']
        test_split_number = 11

class ParisLilleConfig:
    def __init__(self, path):
        self.dataset_path = path
        self.cache_dir = self.dataset_path + '/cache'
        self.test_result_folder = self.dataset_path + '/test'
        self.train_dir = self.dataset_path + "/training_10_classes"
        self.val_files = ['Lille2.ply']
        self.test_dir = self.dataset_path + "/test_10_classes"

class Semantic3DConfig:
    def __init__(self, path):
        self.dataset_path = path
        self.cache_dir = self.dataset_path + '/cache/'
        self.use_cache = True
        self.test_result_folder = self.dataset_path + '/test'
        self.val_split = 1

class TorontoConfig:
    def __init__(self, path):
        self.dataset_path = path
        if not self.dataset_path.endswith("/"):
            self.dataset_path = self.dataset_path + "/"
        self.use_cache = True
        self.cache_dir = self.dataset_path + '/cache'
        self.test_result_folder = self.dataset_path + '/test'
        self.train_files = ['L001.ply', 'L003.ply', 'L004.ply']
        self.val_files = ['L002.ply']
        self.test_files = ['L002.ply']

def print_usage_and_exit():
    print("Usage: ml-test.py [kitti|paris|toronto|sematic3d] path/to/dataset")
    exit(0)

def main():
    if len(sys.argv) != 3:
        print_usage_and_exit()

    which = sys.argv[1]
    path = sys.argv[2]

    if which == "kitti":
        cfg = KITTIConfig(path)
        dataset = SemanticKITTI(cfg)
    elif which == "paris":
        cfg = ParisLilleConfig(path)
        dataset = ParisLille3D(cfg)
    elif which == "toronto":
        cfg = TorontoConfig(path)
        dataset = Toronto3D(cfg)
    elif which == "semantic3d":
        cfg = Semantic3DConfig(path)
        dataset = Semantic3D(cfg)
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

if __name__ == "__main__":
    main()
