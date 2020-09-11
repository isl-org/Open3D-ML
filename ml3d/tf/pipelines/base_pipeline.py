import numpy as np
import yaml
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
        self.cfg.logs_dir = join(self.cfg.main_log_dir,
                                 model.__class__.__name__ + '_tf')
        make_dir(self.cfg.logs_dir)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if device == 'cpu':
                    tf.config.set_visible_devices([], 'GPU')
                elif device == 'gpu':
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                else:
                    idx = device.split(':')[1]
                    tf.config.set_visible_devices(gpus[int(idx)], 'GPU')
            except RuntimeError as e:
                print(e)

    def get_loss(self):
        raise NotImplementedError()
