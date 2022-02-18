import numpy as np
import pandas as pd
import os, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath, isdir
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging

from .utils import DataProcessing, get_min_bbox, BEVBox3D
from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class S3DIS(BaseDataset):
    """This class is used to create a dataset based on the S3DIS (Stanford
    Large-Scale 3D Indoor Spaces) dataset, and used in visualizer, training, or
    testing.

    The S3DIS dataset is best used to train models for building indoors.
    """

    def __init__(self,
                 dataset_path,
                 name='S3DIS',
                 task='segmentation',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 class_weights=[
                     3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                     650464, 791496, 88727, 1284130, 229758, 2272837
                 ],
                 num_points=40960,
                 test_area_idx=3,
                 ignored_label_inds=[],
                 ignored_objects=[
                     'wall', 'floor', 'ceiling', 'beam', 'column', 'clutter'
                 ],
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (S3DIS in this case).
            task: One of {segmentation, detection} for semantic segmentation and object detection.
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            class_weights: The class weights to use in the dataset.
            num_points: The maximum number of points to use when splitting the dataset.
            test_area_idx: The area to use for testing. The valid values are 1 through 6.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            ignored_objects: Ignored objects
            test_result_folder: The folder where the test results should be stored.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         task=task,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         test_result_folder=test_result_folder,
                         num_points=num_points,
                         test_area_idx=test_area_idx,
                         ignored_label_inds=ignored_label_inds,
                         ignored_objects=ignored_objects,
                         **kwargs)

        cfg = self.cfg

        assert isdir(dataset_path), f"Invalid dataset path {dataset_path}"

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.test_split = 'Area_' + str(cfg.test_area_idx)

        self.pc_path = join(self.cfg.dataset_path, 'original_pkl')

        if not exists(self.pc_path):
            print("creating dataset")
            self.create_ply_files(self.cfg.dataset_path, self.label_to_names)

        # TODO : if num of ply files < 272, then create.

        self.all_files = glob.glob(
            str(Path(self.cfg.dataset_path) / 'original_pkl' / '*.pkl'))

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
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
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
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

    """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
    """

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
    def create_ply_files(dataset_path, class_names):
        os.makedirs(join(dataset_path, 'original_pkl'), exist_ok=True)
        anno_file = Path(abspath(
            __file__)).parent / '_resources' / 's3dis_annotation_paths.txt'

        anno_file = str(anno_file)
        anno_paths = [line.rstrip() for line in open(anno_file)]
        anno_paths = [Path(dataset_path) / p for p in anno_paths]

        class_names = [val for key, val in class_names.items()]
        label_to_idx = {l: i for i, l in enumerate(class_names)}

        out_format = '.pkl'

        for anno_path in tqdm(anno_paths):
            elems = str(anno_path).split('/')
            save_path = elems[-3] + '_' + elems[-2] + out_format
            save_path = Path(dataset_path) / 'original_pkl' / save_path

            data_list = []
            bboxes = []
            for file in glob.glob(str(anno_path / '*.txt')):
                class_name = Path(file).name.split('_')[0]
                if class_name not in class_names:
                    class_name = 'clutter'

                pc = pd.read_csv(file, header=None,
                                 delim_whitespace=True).values
                labels = np.ones((pc.shape[0], 1)) * label_to_idx[class_name]
                data_list.append(np.concatenate([pc, labels], 1))
                bbox = [class_name] + get_min_bbox(pc)
                bboxes.append(bbox)

            pc_label = np.concatenate(data_list, 0)

            info = [pc_label, bboxes]
            with open(save_path, 'wb') as f:
                pickle.dump(info, f)

    @staticmethod
    def read_bboxes(bboxes, ignored_objects):
        objects = []
        for box in bboxes:
            name = box[0]
            if name in ignored_objects:
                continue
            center = np.array([box[1], box[2], box[3]])
            size = np.array([box[4], box[5], box[6]])  # w, h, l
            yaw = box[7]

            objects.append(Object3d(name, center, size, yaw))

        return objects


class S3DISSplit(BaseDatasetSplit):
    """This class is used to create a split for S3DIS dataset.

    Initialize the class.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        data = pickle.load(open(pc_path, 'rb'))

        pc, bboxes = data
        pc = pc[~np.isnan(pc).any(1)]

        bboxes = self.dataset.read_bboxes(bboxes, self.cfg.ignored_objects)

        points = np.array(pc[:, :3], dtype=np.float32)
        feat = np.array(pc[:, 3:6], dtype=np.float32)
        labels = np.array(pc[:, 6], dtype=np.int32).reshape((-1,))

        data = {
            'point': points,
            'feat': feat,
            'label': labels,
            'bounding_boxes': bboxes
        }

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.pkl', '')

        pc_path = str(pc_path)
        split = self.split
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': split}
        return attr


class Object3d(BEVBox3D):
    """Stores object specific details like bbox coordinates."""

    def __init__(self, name, center, size, yaw):
        super().__init__(center, size, yaw, name, -1.0)

        self.occlusion = 0.0


DATASET._register_module(S3DIS)
