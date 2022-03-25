import numpy as np
import yaml
import torch
from abc import ABC, abstractmethod

from os.path import join, exists, dirname, abspath

# use relative import for being compatible with Open3d main repo
from ...utils import Config, make_dir


class BasePipeline(ABC):
    """Base pipeline class."""

    def __init__(self,
                 model,
                 dataset=None,
                 device='cuda',
                 distributed=False,
                 **kwargs):
        """Initialize.

        Args:
            model: A network model.
            dataset: A dataset, or None for inference model.
            device: 'cuda' or 'cpu'.
            distributed: Whether to use multiple gpus.
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
        self.rng = np.random.default_rng(kwargs.get('seed', None))

        self.distributed = distributed
        if self.distributed and self.name == "SemanticSegmentation":
            raise NotImplementedError(
                "Distributed training not implemented for SemanticSegmentation!"
            )

        self.rank = kwargs.get('rank', 0)

        dataset_name = dataset.name if dataset is not None else ''
        self.cfg.logs_dir = join(
            self.cfg.main_log_dir,
            model.__class__.__name__ + '_' + dataset_name + '_torch')

        if self.rank == 0:
            make_dir(self.cfg.main_log_dir)
            make_dir(self.cfg.logs_dir)

        if device == 'cpu' or not torch.cuda.is_available():
            if distributed:
                raise NotImplementedError(
                    "Distributed training for CPU is not supported yet.")
            self.device = torch.device('cpu')
        else:
            if distributed:
                self.device = torch.device(device)
                print(f"Rank : {self.rank} using device : {self.device}")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device('cuda')

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
