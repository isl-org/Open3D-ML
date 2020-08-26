#!/usr/bin/env python
from ml3d.datasets import SemanticKITTI
from ml3d.vis import Visualizer
import sys

if len(sys.argv) != 2:
    print("Usage: ml-test.py path/to/kitti/sequences")
    exit(0)

class KITTIConfig:
    def __init__(self, path):
        self.dataset_path = path
        self.training_split = ['01']
        validation_split = ['08']
        test_split_number = 11
        class_weights = [ 55437630, 320797, 541736, 2578735, 3274484, 552662,
                          184064, 78858, 240942562, 17294618, 170599734, 6369672,
                          230413074, 101130274, 476491114, 9833174, 129609852,
                          4506626, 1168181 ]

cfg = KITTIConfig(sys.argv[1])
kitti = SemanticKITTI(cfg)
data = kitti.get_split('training')

#Visualizer.visualize(data, [0, 2, 4, 6])

path2idx = {}
for i in range(0, len(data.path_list)):
    path2idx[data.path_list[i]] = i
indices = [path2idx[p] for p in sorted(path2idx.keys())]
Visualizer.visualize(data, indices[:4])

