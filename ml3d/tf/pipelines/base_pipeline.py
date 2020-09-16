import numpy as np
import yaml
import tensorflow as tf

from os.path import join, exists, dirname, abspath

from ...utils import Config, make_dir


class BasePipeline(object):
    """
    Base pipeline class
    """

    def __init__(self, model, dataset=None, **kwargs):
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
        if kwargs['name'] is None:
            raise KeyError("Please give a name to the pipeline")

        self.cfg = Config(kwargs)
        self.name = self.cfg.name

        self.model = model
        self.dataset = dataset

        make_dir(self.cfg.main_log_dir)
        dataset_name = dataset.name if dataset is not None else ''
        self.cfg.logs_dir = join(
            self.cfg.main_log_dir,
            model.__class__.__name__ + '_' + dataset_name + '_torch')
        make_dir(self.cfg.logs_dir)

    def get_loss(self):
        raise NotImplementedError()
