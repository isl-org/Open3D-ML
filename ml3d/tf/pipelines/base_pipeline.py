import numpy as np
from abc import ABC, abstractmethod
from os.path import join

from ...utils import Config, make_dir


class BasePipeline(ABC):
    """Base pipeline class."""

    def __init__(self, model, dataset=None, **kwargs):
        """Initialize.

        Args:
            model: network
            dataset: dataset, or None for inference model
            device: 'gpu' or 'cpu'
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
        self.rng = np.random.default_rng(kwargs.get('seed', None))

        make_dir(self.cfg.main_log_dir)
        dataset_name = dataset.name if dataset is not None else ''
        self.cfg.logs_dir = join(
            self.cfg.main_log_dir,
            model.__class__.__name__ + '_' + dataset_name + '_tf')
        make_dir(self.cfg.logs_dir)
        self.summary = {}
        self.cfg.setdefault('summary', {})

    @abstractmethod
    def run_inference(self, data):
        """Run inference on a given data.

        Args:
            data: A raw data.

        Returns:
            Returns the inference results.
        """
        return

    @abstractmethod
    def run_test(self):
        """Run testing on test sets."""
        return

    @abstractmethod
    def run_train(self):
        """Run training on train sets."""
        return
