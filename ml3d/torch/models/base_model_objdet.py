import numpy as np
import yaml
import torch
from os.path import join, exists, dirname, abspath
from abc import ABC, abstractmethod

# use relative import for being compatible with Open3d main repo
from ...utils import Config


class BaseModel(ABC, torch.nn.Module):
    """Base dataset class."""

    def __init__(self, **kwargs):
        """Initialize.

        Args:
            cfg (cfg object or str): cfg object or path to cfg file
            dataset_path (str): Path to the dataset
            **kwargs (dict): Dict of args
        """
        super().__init__()

        self.cfg = Config(kwargs)
        self.rng = np.random.default_rng(kwargs.get('seed', None))

    @abstractmethod
    def get_loss(self, results, inputs):
        """Computes the loss given the network input and outputs.

        Args:
            Loss: A loss object.
            results: This is the output of the model.
            inputs: This is the input to the model.

        Returns:
            Returns the loss value.
        """
        return

    @abstractmethod
    def get_optimizer(self, cfg_pipeline):
        """Returns an optimizer object for the model.

        Args:
            cfg_pipeline: A Config object with the configuration of the pipeline.

        Returns:
            Returns a new optimizer object.
        """
        return

    @abstractmethod
    def preprocess(self, cfg_pipeline):
        """Data preprocessing function.

        This function is called before training to preprocess the data from a
        dataset.

        Args:
            data: A sample from the dataset.
            attr: The corresponding attributes.

        Returns:
            Returns the preprocessed data
        """
        return

    @abstractmethod
    def transform(self, cfg_pipeline):
        """Transform function for the point cloud and features.

        Args:
            cfg_pipeline: config file for pipeline.
        """
        return

    @abstractmethod
    def inference_end(self, results, attr=None):
        """This function is called after the inference.

        This function can be implemented to apply post-processing on the
        network outputs.

        Args:
            results: The model outputs as returned by the call() function.
                Post-processing is applied on this object.

        Returns:
            Returns True if the inference is complete and otherwise False.
            Returning False can be used to implement inference for large point
            clouds which require multiple passes.
        """
        return
