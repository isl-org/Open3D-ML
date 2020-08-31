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
        self.cfg = Config.load_from_file(cfg_path)

  
        if cfg is not None:
            if isinstance(cfg, str):
                self.cfg = Config.load_from_file(cfg)
            elif isinstance(cfg, Config):
                self.cfg = cfg
            elif isinstance(cfg, dict):
                self.cfg = self.cfg.merge_from_dict(cfg)
            else:
                raise TypeError("cfg must be a string or Config " +
                                "but got {}".format(type(cfg)))

        self.cfg.merge_from_dict(kwargs)

        if dataset_path is not None:
            self.cfg.dataset_path = dataset_path


    def get_split(self, split):
        raise NotImplementedError()
