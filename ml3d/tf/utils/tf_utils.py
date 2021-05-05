import re
import tensorflow as tf


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def gen_CNN(channels,
            conv=tf.keras.layers.Conv1D,
            use_bias=True,
            activation=tf.keras.layers.ReLU,
            batch_norm=False):
    layers = []
    for i in range(len(channels) - 1):
        in_size, out_size = channels[i:i + 2]
        layers.append(
            conv(out_size, 1, use_bias=use_bias, data_format="channels_first"))
        if batch_norm:
            layers.append(
                tf.keras.layers.BatchNormalization(axis=1,
                                                   momentum=0.9,
                                                   epsilon=1e-05))
        if activation is not None:
            layers.append(activation())

    return tf.keras.Sequential(layers)
