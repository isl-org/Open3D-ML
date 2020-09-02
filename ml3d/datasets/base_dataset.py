import yaml

from os.path import join, exists, dirname, abspath
from ..utils import Config

class BaseDataset(object):
    """
    Base dataset class
    """
    def __init__(self, cfg=None, dataset_path=None, **kwargs):
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
        if dataset_path is None and cfg is None:
            raise KeyError(
            "should specify dataset_path or cfg to initialize a Dataset")

        cfg_path = dirname(abspath(__file__)) + \
                    "/../configs/default_cfgs/" + self.default_cfg_name

        self.cfg = Config.merge_default_cfgs(
                    cfg_path, 
                    cfg, 
                    dataset_path=dataset_path,
                    **kwargs)
  


    def get_split(self, split):
        raise NotImplementedError()
