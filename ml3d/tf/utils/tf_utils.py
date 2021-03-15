import os
import re
import tensorflow as tf


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]
    

def gen_CNN(channels,
            conv=tf.keras.layers.Conv1d,
            bias=True,
            activation=tf.keras.layers.ReLU,
            batch_norm=None,
            instance_norm=None):
    layers = []
    for i in range(len(channels) - 1):
        in_size, out_size = channels[i:i + 2]
        layers.append(conv(in_size, out_size, 1, bias=bias))
        if batch_norm is not None:
            layers.append(batch_norm(out_size))
        if activation is not None:
            layers.append(activation())
        if instance_norm is not None:
            layers.append(
                instance_norm(out_size, affine=False,
                              track_running_stats=False))

    return tf.keras.Sequential(*layers)
