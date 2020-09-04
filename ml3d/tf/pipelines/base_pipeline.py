import numpy as np
import yaml

from ...utils import make_dir
from os.path import join, exists, dirname, abspath

from ...utils import Config

class BasePipeline(object):
    """
    Base dataset class
    """
    def __init__(self,
                model,
                dataset=None, 
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



        self.cfg = Config(kwargs)
        self.name = self.cfg.name

        self.model = model
        self.dataset = dataset

        make_dir(self.cfg.main_log_dir)
        self.cfg.logs_dir = join(self.cfg.main_log_dir, 
                        model.__class__.__name__ + '_torch')
        make_dir(self.cfg.logs_dir)

        
        # self.device = torch.device('cuda' if torch.cuda.is_available() 
        #                             and device == 'gpu' else 'cpu')

    def get_loss(self):
        raise NotImplementedError()
