import numpy as np
import yaml
import torch
from os.path import join, exists, dirname, abspath

# use relative import for being compatible with Open3d main repo
from ...utils import Config, make_dir


class BasePipeline(object):
    """
    Base pipeline class
    """

    def __init__(self, model, dataset=None, device='gpu', **kwargs):
        """
        Initialize
        Args:
            model: network
            dataset: dataset, or None for inference model
            devce: 'gpu' or 'cpu' 
            kwargs:
        Returns:
            class: The corresponding class.
        """

        self.cfg = Config(kwargs)

        if kwargs['name'] is None:
            raise KeyError("Please give a name to the pipeline")
        self.name = self.cfg.name

        self.model = model
        self.dataset = dataset

        make_dir(self.cfg.main_log_dir)
        dataset_name = dataset.name if dataset is not None else ''
        self.cfg.logs_dir = join(self.cfg.main_log_dir,
            model.__class__.__name__ + '_' + dataset_name + '_torch')
        make_dir(self.cfg.logs_dir)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and device == 'gpu' else 'cpu')

    def get_loss(self):
        raise NotImplementedError()
