import os
import re
from torch import nn
import torch.nn.functional as F


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def latest_torch_ckpt(train_ckpt_dir):
    files = os.listdir(train_ckpt_dir)
    ckpt_list = [f for f in files if f.endswith('.pth')]
    if len(ckpt_list) == 0:
        return None
    ckpt_list.sort(key=natural_keys)

    ckpt_name = ckpt_list[-1]
    return os.path.join(train_ckpt_dir, ckpt_name)


def gen_CNN(channels,
            conv=nn.Conv1d,
            bias=True,
            activation=nn.ReLU,
            batch_norm=None,
            instance_norm=None):
    layers = []
    for i in range(len(channels) - 1):
        in_size, out_size = channels[i:i + 2]
        layers.append(conv(in_size, out_size, 1, bias=bias))
        if batch_norm is not None:
            layers.append(batch_norm(out_size))
        if activation is not None:
            layers.append(activation(inplace=True))
        if instance_norm is not None:
            layers.append(
                instance_norm(out_size, affine=False,
                              track_running_stats=False))

    return nn.Sequential(*layers)
