import numpy as np
import yaml
import tensorflow as tf

from os.path import join, exists, dirname, abspath
from pathlib import Path

from ...utils import Config, make_dir, get_tb_hash


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
            model.__class__.__name__ + '_' + dataset_name + '_tf')
        make_dir(self.cfg.logs_dir)

        tensorboard_dir = join(
            self.cfg.train_sum_dir,
            model.__class__.__name__ + '_' + dataset_name + '_tf')
        hsh = get_tb_hash(tensorboard_dir)
        self.tensorboard_dir = join(self.cfg.train_sum_dir,
                                    str(hsh) + '_' + Path(tensorboard_dir).name)

    def get_loss(self):
        raise NotImplementedError()
