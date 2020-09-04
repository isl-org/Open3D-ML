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
            cfg (cfg object or str): cfg object or path to cfg file
            dataset_path (str): path to the dataset
            args (dict): dict of args 
            kwargs:
        Returns:
            class: The corresponding class.
        """
        if kwargs['dataset_path'] is None:
            raise KeyError(
            "should specify dataset_path or cfg to initialize a Dataset")

        self.cfg = Config(kwargs)
        self.name = self.cfg.name
  

    def get_split(self, split):
        raise NotImplementedError()
