import yaml

from os.path import join, exists, dirname, abspath
from ..utils import Config

class BaseDataset(object):
    """
    Base dataset class
    """
    def __init__(self, **kwargs):
        """
        Initialize
        Args:
            dataset_path (str): path to the dataset
            kwargs:
        Returns:
            class: The corresponding class.
        """
        if kwargs['dataset_path'] is None:
            raise KeyError(
            "Please specify dataset_path to initialize a Dataset")

        if kwargs['name'] is None:
            raise KeyError(
            "Please give a name to the dataset")


        self.cfg = Config(kwargs)
        self.name = self.cfg.name
  

    def get_split(self, split):
        raise NotImplementedError()
