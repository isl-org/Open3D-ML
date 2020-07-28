import numpy as np
import pandas as pd
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
from tqdm import tqdm

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
        self.create_ply_files(self.dataset_path, self.label_to_names)

        # print(Path(self.dataset_path) / 'original_ply' / '*.ply')
        # self.all_files = glob.glob(Path(self.path) / 'original_ply' / '*.ply')

    @staticmethod
    def write_ply(filename, field_list, field_names, triangular_faces=None):
        """
        Write ".ply" files
        Parameters
        ----------
        filename : string
            the name of the file to which the data is saved. A '.ply' extension will be appended to the 
            file name if it does no already have one.
        field_list : list, tuple, numpy array
            the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
            tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
            as one field. 
        field_names : list
            the name of each fields as a list of strings. Has to be the same length as the number of 
            fields.
        Examples
        --------
        >>> points = np.random.rand(10, 3)
        >>> write_ply('example1.ply', points, ['x', 'y', 'z'])
        >>> values = np.random.randint(2, size=10)
        >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])
        >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
        >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
        >>> write_ply('example3.ply', [points, colors, values], field_names)
        """

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

    @staticmethod
    def create_ply_files(dataset_path, class_names):
        os.makedirs(Path(dataset_path) / 'original_ply', exist_ok=True)
        anno_file = Path(abspath(__file__)).parent / 'meta' / 's3dis_annotation_paths.txt'
        anno_paths = [line.rstrip() for line in open(anno_file)]
        anno_paths = [Path(dataset_path) / p for p in anno_paths]

        class_names = [val for key, val in class_names.items()]
        label_to_idx = {l: i for i, l in enumerate(class_names)}

        out_format = '.ply' # TODO : Use from config.

        for anno_path in tqdm(anno_paths):
            elems = str(anno_path).split('/')
            save_path = elems[-3] + '_' + elems[-2] + out_format
            save_path = Path(dataset_path) / 'original_ply' / save_path
            print(save_path)

            data_list = []
            for file in glob.glob(str(anno_path / '*.txt')):
                class_name = Path(file).name.split('_')[0]
                if class_name not in class_names:
                    class_name = 'clutter'
                # print(class_name)

                pc = pd.read_csv(file, header=None, delim_whitespace=True).values
                labels = np.ones((pc.shape[0], 1)) * label_to_idx[class_name]
                data_list.append(np.concatenate([pc, labels], 1))
            
            pc_label = np.concatenate(data_list, 0)
            xyz_min = np.amin(pc_label[:, 0:3], axis = 0)
            pc_label[:, 0:3] -= xyz_min

            xyz = pc_label[:, :3].astype(np.float32)
            colors = pc_label[:, 3:6].astype(np.uint8)
            labels = pc_label[:, 6].astype(np.uint8)

            S3DIS.write_ply(str(save_path), (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])




if __name__ == '__main__':
    a = S3DIS()
    # print(a.num_classes)
    # print(a.label_values)
    # print(a.label_to_idx)