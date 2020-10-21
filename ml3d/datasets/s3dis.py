import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging

from .utils import DataProcessing
from .base_dataset import BaseDataset
from ..utils import make_dir, DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class S3DIS(BaseDataset):
    """
    S3DIS dataset, used in visualizer, training, or test
    """

    def __init__(self,
                 dataset_path,
                 name='S3DIS',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 class_weights=[
                     3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                     650464, 791496, 88727, 1284130, 229758, 2272837
                 ],
                 num_points=40960,
                 test_area_idx=3,
                 ignored_label_inds=[],
                 test_result_folder='./test',
                 **kwargs):
        """
        Initialize
        Args:
            dataset_path (str): path to the dataset
            kwargs:
        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         test_result_folder=test_result_folder,
                         num_points=num_points,
                         test_area_idx=test_area_idx,
                         ignored_label_inds=ignored_label_inds,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.test_split = 'Area_' + str(cfg.test_area_idx)

        self.pc_path = join(self.cfg.dataset_path, 'original_ply')

        if not exists(self.pc_path):
            print("creating dataset")
            self.create_ply_files(self.cfg.dataset_path, self.label_to_names)

        # TODO : if num of ply files < 272, then create.

        self.all_files = glob.glob(
            str(Path(self.cfg.dataset_path) / 'original_ply' / '*.ply'))

    @staticmethod
    def get_label_to_names():
        label_to_names = {
            0: 'ceiling',
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
            12: 'clutter'
        }
        return label_to_names

    def get_split(self, split):
        return S3DISSplit(self, split=split)

    def get_split_list(self, split):
        cfg = self.cfg
        dataset_path = cfg.dataset_path
        file_list = []

        if split in ['test', 'testing', 'val', 'validation']:
            file_list = [
                f for f in self.all_files
                if 'Area_' + str(cfg.test_area_idx) in f
            ]
        elif split in ['train', 'training']:
            file_list = [
                f for f in self.all_files
                if 'Area_' + str(cfg.test_area_idx) not in f
            ]
        elif split in ['all']:
            file_list = self.all_files
        else:
            raise ValueError("Invalid split {}".format(split))

        return file_list

    def get_data(self, file_path, is_test=False):
        file_path = Path(file_path)
        kdtree_path = Path(
            file_path
        ).parent.parent / 'cache' / 'KDTree' / file_path.name.replace(
            ".ply", ".pkl")

        with open(kdtree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)

        pc_feat_labels_path = kdtree_path.parent.parent / 'sub' / file_path.name.replace(
            ".ply", "_sub.npy")
        pc_feat_labels = np.load(pc_feat_labels_path)

        feat = pc_feat_labels[:, 3:6]

        if (is_test):
            labels = np.zeros(np.shape(points)[0], dtype=np.uint8)
        else:
            labels = pc_feat_labels[:, 6]

        return points, feat, search_tree, labels

    def is_tested(self, attr):
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        cfg = self.cfg
        name = attr['name'].split('.')[0]
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(pred)

        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(path, self.name, name + '.npy')
        make_dir(Path(store_path).parent)
        np.save(store_path, pred)
        log.info("Saved {} in {}.".format(name, store_path))

    @staticmethod
    def write_ply(filename, field_list, field_names, triangular_faces=None):
        # Format list input to the right form
        field_list = list(field_list) if (type(field_list) == list or
                                          type(field_list) == tuple) else list(
                                              (field_list,))
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
                header.append('element face {:d}'.format(
                    triangular_faces.shape[0]))
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
                type_list = [('k', 'uint8')
                            ] + [(str(ind), 'int32') for ind in range(3)]
                data = np.empty(triangular_faces.shape[0], dtype=type_list)
                data['k'] = np.full((triangular_faces.shape[0],),
                                    3,
                                    dtype=np.uint8)
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
                lines.append('property %s %s' %
                             (field.dtype.name, field_names[i]))
                i += 1

        return lines

    @staticmethod
    def create_ply_files(dataset_path, class_names):
        os.makedirs(join(dataset_path, 'original_ply'), exist_ok=True)
        anno_file = Path(abspath(
            __file__)).parent / '_resources' / 's3dis_annotation_paths.txt'
        print(anno_file)
        anno_file = str(anno_file)
        anno_paths = [line.rstrip() for line in open(anno_file)]
        anno_paths = [Path(dataset_path) / p for p in anno_paths]

        class_names = [val for key, val in class_names.items()]
        label_to_idx = {l: i for i, l in enumerate(class_names)}

        out_format = '.ply'  # TODO : Use from config.

        for anno_path in tqdm(anno_paths):
            elems = str(anno_path).split('/')
            save_path = elems[-3] + '_' + elems[-2] + out_format
            save_path = Path(dataset_path) / 'original_ply' / save_path

            data_list = []
            for file in glob.glob(str(anno_path / '*.txt')):
                class_name = Path(file).name.split('_')[0]
                if class_name not in class_names:
                    class_name = 'clutter'

                pc = pd.read_csv(file, header=None,
                                 delim_whitespace=True).values
                labels = np.ones((pc.shape[0], 1)) * label_to_idx[class_name]
                data_list.append(np.concatenate([pc, labels], 1))

            pc_label = np.concatenate(data_list, 0)

            xyz = pc_label[:, :3].astype(np.float32)
            colors = pc_label[:, 3:6].astype(np.uint8)
            labels = pc_label[:, 6].astype(np.uint8)

            S3DIS.write_ply(str(save_path), (xyz, colors, labels),
                            ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])


class S3DISSplit():

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        data = PlyData.read(pc_path)['vertex']

        points = np.zeros((data['x'].shape[0], 3), dtype=np.float32)
        points[:, 0] = data['x']
        points[:, 1] = data['y']
        points[:, 2] = data['z']

        feat = np.zeros(points.shape, dtype=np.float32)
        feat[:, 0] = data['red']
        feat[:, 1] = data['green']
        feat[:, 2] = data['blue']

        labels = np.array(data['class'], dtype=np.int32).reshape((-1,))

        data = {'point': points, 'feat': feat, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.ply', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}
        return attr


DATASET._register_module(S3DIS)
