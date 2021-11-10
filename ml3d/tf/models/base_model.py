import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

# use relative import for being compatible with Open3d main repo
from ...utils import Config


class BaseModel(ABC, tf.keras.Model):
    """Base class for models.

    All models must inherit from this class and implement all functions to be
    used with a pipeline.

    Args:
        **kwargs: Configuration of the model as keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = Config(kwargs)
        self.rng = np.random.default_rng(kwargs.get('seed', None))

    def get_loss(self, Loss, results, inputs):
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
    def preprocess(self, data, attr):
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
    def transform(self, *args):
        """Transform function for the point cloud and features.

        Args:
            args: A list of tf Tensors.
        """
        return []

    @abstractmethod
    def inference_begin(self, data):
        """Function called right before running inference.

        Args:
            data: A data from the dataset.
        """
        return

    @abstractmethod
    def inference_preprocess(self):
        """This function prepares the inputs for the model.

        Returns:
            The inputs to be consumed by the call() function of the model.
        """
        return

    @abstractmethod
    def inference_end(self, results):
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
