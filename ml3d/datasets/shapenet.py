import os
from os.path import exists, join
from pathlib import Path
import logging
import json

import numpy as np

from .base_dataset import BaseDataset
from ..utils import make_dir, DATASET

log = logging.getLogger(__name__)


class ShapeNet(BaseDataset):
    """This class is used to create a dataset based on the ShapeNet dataset, and
    used in object detection, visualizer, training, or testing.

    The ShapeNet dataset includes a large set of 3D shapes.
    """

    def __init__(self,
                 dataset_path,
                 name="ShapeNet",
                 class_weights=[
                     2690, 76, 55, 1824, 3746, 69, 787, 392, 1546, 445, 202,
                     184, 275, 66, 152, 5266
                 ],
                 ignored_label_inds=[],
                 test_result_folder='./test',
                 task="classification",
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (ShapeNet in this case).
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.
            task: The task that identifies the purpose. The valid values are classification and segmentation.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir='./logs/cache',
                         use_cache=False,
                         task=task,
                         class_weights=class_weights,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         **kwargs)

        assert task in ['classification',
                        'segmentation'], f"Invalid task {task}"

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.dataset_path = join(
            dataset_path, 'shapenetcore_partanno_segmentation_benchmark_v0')
        self.task = task

        self.cat = {}
        self.catfile = os.path.join(self.dataset_path,
                                    'synsetoffset2category.txt')
        with open(self.catfile, 'r') as f:
            for idx, line in enumerate(f):
                ls = line.strip().split()
                self.cat[idx] = ls[1]

        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.dataset_path, self.cat[item],
                                     'points')
            dir_seg = os.path.join(self.dataset_path, self.cat[item],
                                   'points_label')
            fns = sorted(os.listdir(dir_point))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(
                    (join(dir_point,
                          token + '.pts'), join(dir_seg, token + '.seg')))

        splits = []
        splits_path = join(self.dataset_path, 'train_test_split')
        for split in [
                'shuffled_train_file_list.json', 'shuffled_test_file_list.json',
                'shuffled_val_file_list.json'
        ]:
            with open(join(splits_path, split)) as source:
                json_source = source.read()
                splits.append(
                    [i.split('/')[-1] for i in json.loads(json_source)])
        train_split, test_split, val_split = splits

        self.all_files = []
        self.train_files = []
        self.val_files = []
        self.test_files = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.all_files.append((item, fn[0], fn[1]))
                file = fn[0].split('/')[-1].split('.')[0]
                if file in train_split:
                    self.train_files.append((item, fn[0], fn[1]))
                elif file in test_split:
                    self.test_files.append((item, fn[0], fn[1]))
                else:
                    self.val_files.append((item, fn[0], fn[1]))
        self.rng.shuffle(self.train_files)
        self.rng.shuffle(self.test_files)
        self.rng.shuffle(self.val_files)

    @staticmethod
    def get_label_to_names(task="classification"):
        """Returns a label to names dictionary object depending on the task. The
        valid values for task for classification and segmentation.

        Returns:
            A dict where keys are label numbers and values are the corresponding
            names.
        """
        if task == "classification":
            label_to_names = {
                0: 'Airplane',
                1: 'Bag',
                2: 'Cap',
                3: 'Car',
                4: 'Chair',
                5: 'Earphone',
                6: 'Guitar',
                7: 'Knife',
                8: 'Lamp',
                9: 'Laptop',
                10: 'Motorbike',
                11: 'Mug',
                12: 'Pistol',
                13: 'Rocket',
                14: 'Skateboard',
                15: 'Table'
            }
        elif task == "segmentation":
            label_to_names = {}
            for i in range(50):
                label_to_names[i] = f"Part{i}"
        else:
            raise ValueError(f"Invalid task {task}")
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return ShapeNetSplit(self, split=split, task=self.task)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
        """
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        elif split in ['val', 'validation']:
            files = self.val_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
        else:
            raise ValueError(f"Invalid split {split}")
        return files

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.labels')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels'] + 1
        store_path = join(path, self.name, name + '.labels')
        make_dir(Path(store_path).parent)
        np.savetxt(store_path, pred.astype(np.int32), fmt='%d')

        log.info("Saved {} in {}.".format(name, store_path))


class ShapeNetSplit:
    """The class gets data and attributes based on the split and
    classification.
    """

    def __init__(self, dataset, split='training', task='classification'):
        assert task in ['classification',
                        'segmentation'], f"Invalid task {task}"

        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)

        self.path_list = path_list
        self.split = split
        self.dataset = dataset
        self.task = task

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        path = self.path_list[idx]
        points = np.loadtxt(path[1], dtype=np.float32)
        label = np.loadtxt(
            path[2],
            dtype=np.int64) if self.task == 'segmentation' else np.array(
                [np.int64(path[0])])
        return {'point': points, 'feat': None, 'label': label}

    def get_attr(self, idx):
        name = self.path_list[idx][1].split('/')[-1].split('.')[0]
        return {
            'name': name,
            'path': str(Path(self.path_list[idx][1])),
            'split': self.split
        }


DATASET._register_module(ShapeNet)
