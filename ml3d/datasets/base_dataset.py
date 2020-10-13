import yaml
from abc import ABC, abstractmethod
from os.path import join, exists, dirname, abspath

from ..utils import Config


class BaseDataset(ABC):
    """Base dataset class.

    All datasets must inherit from this class and implement all functions to be
    compatible with pipelines.

    Args:
        **kwargs: Configuration of the model as keyword arguments.

    Attributes:
        cfg: The configuration as Config object that stores the keyword
            arguments that were passed to the constructor.
        name: The name of the dataset.
    """

    def __init__(self, **kwargs):
        if kwargs['dataset_path'] is None:
            raise KeyError(
                "Please specify dataset_path to initialize a Dataset")

        if kwargs['name'] is None:
            raise KeyError("Please give a name to the dataset")

        self.cfg = Config(kwargs)
        self.name = self.cfg.name

    @staticmethod
    @abstractmethod
    def get_label_to_names():
        """Returns a label to names dict.
        
        Returns:
            A dict where keys are label numbers and 
            vals are the corresponding names.
        """

    @abstractmethod
    def get_split(self, split):
        """Returns a dataset split.
        
        Args:
            split: A string identifying the dataset split. Usually one of
            'training', 'test', 'validation', 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return

    @abstractmethod
    def is_tested(self, attr):
        """Checks whether a datum has been tested.

        Args:
            attr: The attributes associated with the datum

        Returns:
            True if the test result has been stored for the datum with the
            specified attribute or else returns False.
        """
        return False

    @abstractmethod
    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with 'attr'.
            attr: The attributes that correspond to the outputs 'results'.
        """
        return


class BaseDatasetSplit(ABC):
    """The base class for dataset splits.

    This class provides access to the data of a specified subset of a dataset.

    Args:
        dataset: The dataset object associated to this split.
        split: A string identifying the dataset split. Usually one of
            'training', 'test', 'validation', 'all'.

    Attributes:
        cfg: Shortcut to the config of the dataset object.
        dataset: The dataset object associated to this split.
        split: A string identifying the dataset split. Usually one of
            'training', 'test', 'validation', 'all'.
    """

    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        self.split = split
        self.dataset = dataset

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
        """Returns the attributes for the given index"""
        return {}
