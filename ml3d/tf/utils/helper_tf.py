"""Wrapper functions for TensorFlow layers."""

import numpy as np
import tensorflow as tf


class ExpandDims(tf.keras.layers.Layer):

    def __init__(self, axis=-1):
        super(ExpandDims, self).__init__()
        self.axis = axis

    def build(self, input_shape):
        pass

    def call(self, input):
        return tf.expand_dims(input, axis=self.axis)


class conv2d(tf.keras.layers.Layer):

    def __init__(self,
                 batchNorm,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 activation=True):
        super(conv2d, self).__init__()

        self.conv = tf.keras.layers.Conv2D(out_planes,
                                           kernel_size,
                                           strides=(stride, stride))

        self.batchNorm = batchNorm
        if self.batchNorm:
            self.batch_normalization = tf.keras.layers.BatchNormalization(
                momentum=0.99, epsilon=1e-6)

        if activation:
            self.activation_fn = tf.keras.layers.LeakyReLU(alpha=0.2)
        else:
            self.activation_fn = None

    # def build(self, input_shape):
    #     super(conv2d, self).build(input_shape)
    #     self.conv.build(input_shape)
    #     self._biases  = self.conv.bias
    #     self._weights = self.conv.kernel

    def call(self, x, training=False):
        x = self.conv(x)
        if self.batchNorm:
            x = self.batch_normalization(x, training=training)
        if self.activation_fn:
            x = self.activation_fn(x)

        return x


class conv2d_transpose(tf.keras.layers.Layer):

    def __init__(self,
                 batchNorm,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 activation=True):
        super(conv2d_transpose, self).__init__()

        self.conv = tf.keras.layers.Conv2DTranspose(out_planes,
                                                    kernel_size,
                                                    strides=(stride, stride))

        self.batchNorm = batchNorm
        if self.batchNorm:
            self.batch_normalization = tf.keras.layers.BatchNormalization(
                momentum=0.99, epsilon=1e-6)

        if activation:
            self.activation_fn = tf.keras.layers.LeakyReLU(alpha=0.2)
        else:
            self.activation_fn = None

    # def build(self, input_shape):
    #     super(conv2d_transpose, self).build(input_shape)
    #     self.conv.build(input_shape)
    #     self.biases  = self.conv.bias
    #     self.weights = self.conv.kernel

    def call(self, x, training=False):
        x = self.conv(x)
        if self.batchNorm:
            x = self.batch_normalization(x, training=training)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


def dropout(inputs, is_training, scope, keep_prob=0.5, noise_shape=None):
    """Dropout layer.

    Args:
      inputs: tensor
      is_training: boolean tf.Variable
      scope: string
      keep_prob: float in [0,1]
      noise_shape: list of ints

    Returns:
      tensor variable
    """
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs
