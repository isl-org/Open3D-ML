import os
from glob import glob
import numpy as np
import plyfile # replace with open3d

class ScanNet:
    """ScanNet reader class.

    This class reads the ScanNet point clouds and label information

    Args:
        root_dir: Path to the root dir. This directory contains the 'scans' 
                  directory.

        mode: Either 'train' or 'test'. If 'test' scans will be used from the 
              'scans_test' dir.

        scenes: A list of scene ids, e.g. ['scene0568_00', 'scene0685_02', ...]. 
                This parameter is optional and can be used to define a subset.
    """

    nyu_labels = [
        (0, 'unlabeled'),
        (1, 'wall'),
        (2, 'floor'),
        (3, 'cabinet'),
        (4, 'bed'),
        (5, 'chair'),
        (6, 'sofa'),
        (7, 'table'),
        (8, 'door'),
        (9, 'window'),
        (10, 'bookshelf'),
        (11, 'picture'),
        (12, 'counter'),
        (13, 'blinds'),
        (14, 'desk'),
        (15, 'shelves'),
        (16, 'curtain'),
        (17, 'dresser'),
        (18, 'pillow'),
        (19, 'mirror'),
        (20, 'floor mat'),
        (21, 'clothes'),
        (22, 'ceiling'),
        (23, 'books'),
        (24, 'refridgerator'),
        (25, 'television'),
        (26, 'paper'),
        (27, 'towel'),
        (28, 'shower curtain'),
        (29, 'box'),
        (30, 'whiteboard'),
        (31, 'person'),
        (32, 'nightstand'),
        (33, 'toilet'),
        (34, 'sink'),
        (35, 'lamp'),
        (36, 'bathtub'),
        (37, 'bag'),
        (38, 'otherstructure'),
        (39, 'otherfurniture'),
        (40, 'otherprop'),
        ]
    
    scannet_labels = [
        (0, 'unlabeled'),
        (1, 'wall'),
        (2, 'floor'),
        (3, 'cabinet'),
        (4, 'bed'),
        (5, 'chair'),
        (6, 'sofa'),
        (7, 'table'),
        (8, 'door'),
        (9, 'window'),
        (10, 'bookshelf'),
        (11, 'picture'),
        (12, 'counter'),
        (14, 'desk'),
        (16, 'curtain'),
        (24, 'refridgerator'),
        (28, 'shower curtain'),
        (33, 'toilet'),
        (34, 'sink'),
        (36, 'bathtub'),
        (39, 'otherfurniture')]

    scannet_label_names = [ x[1] for x in scannet_labels ]

    nyu_colors = np.array([[  0,   0,   0],
       [174, 199, 232],
       [152, 223, 138],
       [ 31, 119, 180],
       [255, 187, 120],
       [188, 189,  34],
       [140,  86,  75],
       [255, 152, 150],
       [214,  39,  40],
       [197, 176, 213],
       [148, 103, 189],
       [196, 156, 148],
       [ 23, 190, 207],
       [178,  76,  76],
       [247, 182, 210],
       [ 66, 188, 102],
       [219, 219, 141],
       [140,  57, 197],
       [202, 185,  52],
       [ 51, 176, 203],
       [200,  54, 131],
       [ 92, 193,  61],
       [ 78,  71, 183],
       [172, 114,  82],
       [255, 127,  14],
       [ 91, 163, 138],
       [153,  98, 156],
       [140, 153, 101],
       [158, 218, 229],
       [100, 125, 154],
       [178, 127, 135],
       [120, 185, 128],
       [146, 111, 194],
       [ 44, 160,  44],
       [112, 128, 144],
       [ 96, 207, 209],
       [227, 119, 194],
       [213,  92, 176],
       [ 94, 106, 211],
       [ 82,  84, 163],
       [100,  85, 144]], dtype=np.uint8)

    nyu40_to_scannet20 = np.zeros(41, dtype=np.int32)
    for i, l in enumerate(scannet_labels):
        nyu40_to_scannet20[l[0]] = i
    
    def __init__(self, root_dir, mode='train', scenes=None):
        self.root_dir = root_dir
        self.mode = mode

        self.scene_dirs = []

        if 'train' == mode:
            self.scene_dirs = sorted(glob(os.path.join(root_dir, 'scans', 'scene*_*')))
        elif 'test' == mode:
            self.scene_dirs = sorted(glob(os.path.join(root_dir, 'scans_test', 'scene*_*')))
        else:
            raise Exception("Invalid mode '{}'; mode must be one of ['train', 'test'].".format(mode))

        if not scenes is None:
            scenes_set = set(scenes)
            filter_fn = lambda x: os.path.basename(x) in scenes_set
            self.scene_dirs = list(filter(filter_fn, self.scene_dirs))
            if len(scenes_set) != len(self.scene_dirs):
                diff = scenes_set - set(map(lambda x: os.path.basename(x), self.scene_dirs))
                raise Exception("Not all scenes found. Missing scenes are {}".format(diff))


    @staticmethod
    def load_scene(scene_dir):
        """Returns a dictionary with the points, colors and labels
        
        Args:
            scene_dir: The path to the scene directory.
        """
        scene_id = os.path.basename(scene_dir)

        color_pointcloud_path = os.path.join(scene_dir, '{}_vh_clean_2.ply'.format(scene_id))
        with open(color_pointcloud_path, 'rb') as f:
            plydata = plyfile.PlyData.read(f)
            vertices = np.empty((plydata['vertex'].count, 3), dtype=np.float32)
            vertices[:,0] = plydata['vertex'].data['x']
            vertices[:,1] = plydata['vertex'].data['y']
            vertices[:,2] = plydata['vertex'].data['z']

            colors = np.empty((plydata['vertex'].count, 3), dtype=np.uint8)
            colors[:,0] = plydata['vertex'].data['red']
            colors[:,1] = plydata['vertex'].data['green']
            colors[:,2] = plydata['vertex'].data['blue']


        label_pointcloud_path = os.path.join(scene_dir, '{}_vh_clean_2.labels.ply'.format(scene_id))
        if os.path.exists(label_pointcloud_path):
            with open(label_pointcloud_path, 'rb') as f:
                plydata = plyfile.PlyData.read(f)
                labels = np.asarray(plydata['vertex'].data['label'])
        else:
            labels = None

        return {'scene_id': scene_id, 'points': vertices, 'colors': colors, 'labels': labels}

    def get(self, index):
        """Returns a dictionary with the points, colors and labels
        
        Args:
            index: The integer scene index.
        """
        scene_dir = self.scene_dirs[index]
        return self.load_scene(scene_dir)

    def __getitem__(self, index):
        return self.get(index)
        
    def __len__(self):
        """Returns the number of scenes."""
        return len(self.scene_dirs)
