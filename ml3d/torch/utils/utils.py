import torch
import numpy as np
from os import makedirs
from os.path import exists, join, isfile, dirname, abspath


def make_dir(folder_name):
    if not exists(folder_name):
        makedirs(folder_name)


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

