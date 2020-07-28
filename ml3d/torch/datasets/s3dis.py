import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath

#TODO : move to config
dataset_path = '/Users/sanskara/Downloads/Stanford3dDataset_v1.2_Aligned_Version/'
test_area_idx = 3

class S3DIS:
    def __init__(self):
        self.name = 'S3DIS'
        self.dataset_path = dataset_path
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)

        self.pc_path = Path(self.dataset_path) / 'original_ply'
        # if not exists(self.pc_path):
        self.create_ply_files(self.dataset_path)

        # print(Path(self.dataset_path) / 'original_ply' / '*.ply')
        # self.all_files = glob.glob(Path(self.path) / 'original_ply' / '*.ply')

    @staticmethod
    def create_ply_files(path):
    	os.makedirs(Path(path) / 'original_ply', exist_ok=True)
    	anno_file = Path(abspath(__file__)).parent / 'meta' / 's3dis_annotation_paths.txt'
    	anno_paths = [line.rstrip() for line in open(anno_file)]
    	anno_paths = [Path(path) / p for p in anno_paths]
    	


if __name__ == '__main__':
	a = S3DIS()
	# print(a.num_classes)
	# print(a.label_values)
	# print(a.label_to_idx)