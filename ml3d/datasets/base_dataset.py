import yaml
from abc import ABC, abstractmethod
from os.path import join, exists, dirname, abspath
import logging
import numpy as np

from ..utils import Config, get_module

log = logging.getLogger(__name__)


class BaseDataset(ABC):
    """The base dataset class that is used by all other datasets.

    All datasets must inherit from this class and implement the functions in order to be
    compatible with pipelines.

    Args:
        **kwargs: The configuration of the model as keyword arguments.

    Attributes:
        cfg: The configuration file as Config object that stores the keyword
            arguments that were passed to the constructor.
        name: The name of the dataset.

    **Example:**
        This example shows a custom dataset that inherit from the base_dataset class:

            from .base_dataset import BaseDataset

            class MyDataset(BaseDataset):
            def __init__(self,
                 dataset_path,
                 name='CustomDataset',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[],
                 test_result_folder='./test',
                 val_files=['Custom.ply'],
                 **kwargs):
    """

    def __init__(self, **kwargs):
        """Initialize the class by passing the dataset path."""
        if kwargs['dataset_path'] is None:
            raise KeyError("Provide dataset_path to initialize the dataset")

        if kwargs['name'] is None:
            raise KeyError("Provide dataset name to initialize it")

        self.cfg = Config(kwargs)
        self.name = self.cfg.name
        self.rng = np.random.default_rng(kwargs.get('seed', None))

    @staticmethod
    @abstractmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """

    @abstractmethod
    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return

    @abstractmethod
    def is_tested(self, attr):
        """Checks whether a datum has been tested.

        Args:
            attr: The attributes associated with the datum.

        Returns:
            This returns True if the test result has been stored for the datum with the
            specified attribute; else returns False.
        """
        return False

    @abstractmethod
    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        return


class BaseDatasetSplit(ABC):
    """The base class for dataset splits.

    This class provides access to the data of a specified subset or split of a dataset.

    Args:
        dataset: The dataset object associated to this split.
        split: A string identifying the dataset split, usually one of
            'training', 'test', 'validation', or 'all'.

    Attributes:
        cfg: Shortcut to the config of the dataset object.
        dataset: The dataset object associated to this split.
        split: A string identifying the dataset split, usually one of
            'training', 'test', 'validation', or 'all'.
    """

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        self.path_list = path_list
        self.split = split
        self.dataset = dataset

        if split in ['test']:
            sampler_cls = get_module('sampler', 'SemSegSpatiallyRegularSampler')
        else:
            sampler_cfg = self.cfg.get('sampler',
                                       {'name': 'SemSegRandomSampler'})
            sampler_cls = get_module('sampler', sampler_cfg['name'])
        self.sampler = sampler_cls(self)

    @abstractmethod
    def __len__(self):
        """Returns the number of samples in the split."""
        return 0

    @abstractmethod
    def get_data(self, idx):
        """Returns the data for the given index."""
        return {}

    @abstractmethod
    def get_attr(self, idx):
        """Returns the attributes for the given index."""
        return {}
