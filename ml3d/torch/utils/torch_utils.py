import os
import re


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
