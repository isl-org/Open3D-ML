import numpy as np
import yaml
import torch

from ...utils import make_dir
from os.path import join, exists, dirname, abspath

from ...utils import Config

class BasePipeline(object):
    """
    Base dataset class
    """
    def __init__(self, 
                model=None, 
                dataset=None, 
                cfg=None, 
                device=None,
                **kwargs):
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

        cfg_path = dirname(abspath(__file__)) + \
                    "/../../configs/default_cfgs/" + self.default_cfg_name
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

        self.cfg = self.cfg.merge_from_dict(kwargs)

        self.model = model
        self.dataset = dataset

        make_dir(self.cfg.main_log_dir)
        self.cfg.logs_dir = join(self.cfg.main_log_dir, 
                        model.__class__.__name__ + '_torch')
        make_dir(self.cfg.logs_dir)

        
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                    and device == 'gpu' else 'cpu')

    def get_loss(self):
        raise NotImplementedError()
