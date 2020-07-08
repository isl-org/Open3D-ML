from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import os, glob, pickle
import sys

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

grid_size = 0.06
dataset_path = '/data/semantic3d/original_data'
original_pc_folder = join(dirname(dataset_path), 'original_ply')
sub_pc_folder = join(dirname(dataset_path), 'input_{:.3f}'.format(grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None

for pc_path in glob.glob(join(dataset_path, '*.txt')):
    print(pc_path)
    file_name = pc_path.split('/')[-1][:-4]

    # check if it has already calculated
    if exists(join(sub_pc_folder, file_name + '_KDTree.pkl')):
        continue

    pc = DP.load_pc_semantic3d(pc_path)
    # check if label exists
    label_path = pc_path[:-4] + '.labels'
    if exists(label_path):
        labels = DP.load_label_semantic3d(label_path)
        full_ply_path = join(original_pc_folder, file_name + '.ply')

        #  Subsample to save space
        sub_points, sub_colors, sub_labels = DP.grid_sub_sampling(pc[:, :3].astype(np.float32),
                                                                  pc[:, 4:7].astype(np.uint8), labels, 0.01)
        sub_labels = np.squeeze(sub_labels)

        write_ply(full_ply_path, (sub_points, sub_colors, sub_labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        # save sub_cloud and KDTree file
        sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(sub_points, sub_colors, sub_labels, grid_size)
        sub_colors = sub_colors / 255.0
        sub_labels = np.squeeze(sub_labels)
        sub_ply_file = join(sub_pc_folder, file_name + '.ply')
        write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

        search_tree = KDTree(sub_xyz, leaf_size=50)
        kd_tree_file = join(sub_pc_folder, file_name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        proj_idx = np.squeeze(search_tree.query(sub_points, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(sub_pc_folder, file_name + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, labels], f)

    else:
        full_ply_path = join(original_pc_folder, file_name + '.ply')
        write_ply(full_ply_path, (pc[:, :3].astype(np.float32), pc[:, 4:7].astype(np.uint8)),
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        # save sub_cloud and KDTree file
        sub_xyz, sub_colors = DP.grid_sub_sampling(pc[:, :3].astype(np.float32), pc[:, 4:7].astype(np.uint8),
                                                   grid_size=grid_size)
        sub_colors = sub_colors / 255.0
        sub_ply_file = join(sub_pc_folder, file_name + '.ply')
        write_ply(sub_ply_file, [sub_xyz, sub_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])
        labels = np.zeros(pc.shape[0], dtype=np.uint8)

        search_tree = KDTree(sub_xyz, leaf_size=50)
        kd_tree_file = join(sub_pc_folder, file_name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        proj_idx = np.squeeze(search_tree.query(pc[:, :3].astype(np.float32), return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(sub_pc_folder, file_name + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, labels], f)
