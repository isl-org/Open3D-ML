import numpy as np
import yaml
import torch
from os.path import join, exists, dirname, abspath

# use relative import for being compatible with Open3d main repo 
from ...utils import Config

class BaseModel(torch.nn.Module):
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
        super().__init__()
 
        self.cfg = Config(kwargs)



    def get_loss(self):
        raise NotImplementedError()
