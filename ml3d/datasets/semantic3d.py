import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from tqdm import tqdm
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging

from .utils import DataProcessing as DP

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class Semantic3DSplit():
    def __init__(self, dataset, split='training'):
        self.cfg    = dataset.cfg
        path_list   = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))
        # if split == 'test':
        #     dataset.test_list = path_list
        #     for test_file_name in path_list:
        #         points = np.load(test_file_name)
        #         dataset.possibility += [np.random.rand(points.shape[0]) * 1e-3]
        #         dataset.min_possibility += [float(np.min(dataset.possibility[-1]))]
                
        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        log.debug("get_data called {}".format(pc_path))

        pc = pd.read_csv(pc_path, header=None, delim_whitespace=True, dtype = np.float32).values

        points = pc[:, 0:3]
        feat = pc[:, [4, 5, 6, 3]]


        if(self.split != 'test'):
            labels = pd.read_csv(pc_path.replace(".txt", ".labels"), header=None, delim_whitespace=True, dtype = np.int32).values
        else:
            labels = np.zeros((points.shape[0], ), dtype = np.int32)
        
        data = {
            'point' : points,
            'feat' : feat,
            'label' : labels
        }

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        attr = {
            'name'      : name,
            'path'      : str(pc_path),
            'split'     : self.split
        }
        return attr


class Semantic3D:
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = 'Semantic3D'
        self.dataset_path = cfg.dataset_path
        self.label_to_names = {0: 'unlabeled',
                               1: 'man-made terrain',
                               2: 'natural terrain',
                               3: 'high vegetation',
                               4: 'low vegetation',
                               5: 'buildings',
                               6: 'hard scape',
                               7: 'scanning artefacts',
                               8: 'cars'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        self.all_files = glob.glob(str(Path(self.dataset_path) / '*.txt'))
        random.shuffle(self.all_files)

        self.train_files = [f for f in self.all_files if exists(Path(f).parent / Path(f).name.replace('.txt', '.labels'))]
        self.test_files = [f for f in self.all_files if f not in self.train_files]

        self.all_split = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.val_split = cfg.val_split

        self.train_files = np.sort(self.train_files)
        self.test_files = np.sort(self.test_files)
        self.val_files = []

        for i, file_path in enumerate(self.train_files):
            if self.all_split[i] == self.val_split:
                self.val_files.append(file_path)

        self.train_files = np.sort([f for f in self.train_files if f not in self.val_files])

    def get_split (self, split):
        return Semantic3DSplit(self, split=split)
    
    def get_split_list(self, split):
        if split == 'test':
            random.shuffle(self.test_files)
            return self.test_files
        elif split == 'training':
            random.shuffle(self.train_files)
            return self.train_files
        else:
            random.shuffle(self.val_files)
            return self.val_files

    def crop_pc(self, points, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        if(points.shape[0] < self.cfg.num_points):
            select_idx = np.array(range(points.shape[0]))
            diff = self.cfg.num_points - points.shape[0]
            select_idx = list(select_idx) + list(random.choices(select_idx, k = diff))
            random.shuffle(select_idx)
            return select_idx
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, 
                                k=self.cfg.num_points)[1][0]

        select_idx = DP.shuffle_idx(select_idx)
        return select_idx


    @staticmethod
    def write_ply(filename, field_list, field_names, triangular_faces=None):
        # Format list input to the right form
        field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
        for i, field in enumerate(field_list):
            if field.ndim < 2:
                field_list[i] = field.reshape(-1, 1)
            if field.ndim > 2:
                print('fields have more than 2 dimensions')
                return False    

        # check all fields have the same number of data
        n_points = [field.shape[0] for field in field_list]
        if not np.all(np.equal(n_points, n_points[0])):
            print('wrong field dimensions')
            return False    

        # Check if field_names and field_list have same nb of column
        n_fields = np.sum([field.shape[1] for field in field_list])
        if (n_fields != len(field_names)):
            print('wrong number of field names')
            return False

        # Add extension if not there
        if not filename.endswith('.ply'):
            filename += '.ply'

        # open in text mode to write the header
        with open(filename, 'w') as plyfile:

            # First magical word
            header = ['ply']

            # Encoding format
            header.append('format binary_' + sys.byteorder + '_endian 1.0')

            # Points properties description
            header.extend(S3DIS.header_properties(field_list, field_names))

            # Add faces if needded
            if triangular_faces is not None:
                header.append('element face {:d}'.format(triangular_faces.shape[0]))
                header.append('property list uchar int vertex_indices')

            # End of header
            header.append('end_header')

            # Write all lines
            for line in header:
                plyfile.write("%s\n" % line)

        # open in binary/append to use tofile
        with open(filename, 'ab') as plyfile:

            # Create a structured array
            i = 0
            type_list = []
            for fields in field_list:
                for field in fields.T:
                    type_list += [(field_names[i], field.dtype.str)]
                    i += 1
            data = np.empty(field_list[0].shape[0], dtype=type_list)
            i = 0
            for fields in field_list:
                for field in fields.T:
                    data[field_names[i]] = field
                    i += 1

            data.tofile(plyfile)

            if triangular_faces is not None:
                triangular_faces = triangular_faces.astype(np.int32)
                type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
                data = np.empty(triangular_faces.shape[0], dtype=type_list)
                data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
                data['0'] = triangular_faces[:, 0]
                data['1'] = triangular_faces[:, 1]
                data['2'] = triangular_faces[:, 2]
                data.tofile(plyfile)

        return True

    @staticmethod
    def header_properties(field_list, field_names):
        # List of lines to write
        lines = []

        # First line describing element vertex
        lines.append('element vertex %d' % field_list[0].shape[0])

        # Properties lines
        i = 0
        for fields in field_list:
            for field in fields.T:
                lines.append('property %s %s' % (field.dtype.name, field_names[i]))
                i += 1

        return lines


from ml3d.utils import Config


if __name__ == '__main__':
    config = '../configs/randlanet_semantic3d.py'
    cfg  = Config.load_from_file(config)
    a = Semantic3D(cfg.dataset)
    b = a.get_split("test")
    c = b.get_data(1)
    print(b.get_attr(1)['name'])
    print(c['point'].shape)
    print(c['feat'].shape)
    print(c['label'].shape)
    print(c['point'][0])
    print(c['feat'][0])
    print(c['label'][0])
    print(c['label'].mean())
    # print(c['label'])
    # # print(b.get_data(0))
    # print(b.get_attr(10))