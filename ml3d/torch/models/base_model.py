import numpy as np
import yaml
import torch

from os.path import join, exists, dirname, abspath

from ...utils import Config

class BaseModel(torch.nn.Module):
    """
    Base dataset class
    """
    def __init__(self, cfg=None, args=None, **kwargs):
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
        super().__init__()
 
        cfg_path = dirname(abspath(__file__)) + \
                    "/../../configs/default_cfgs/" + self.default_cfg_name
        
        self.cfg = Config.merge_default_cfgs(
                    cfg_path, 
                    cfg, 
                    **kwargs)



    def get_loss(self):
        raise NotImplementedError()
