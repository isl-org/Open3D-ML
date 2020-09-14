import torch
import hashlib
from pathlib import Path
from typing import Callable
import numpy as np

from os import makedirs, listdir
from os.path import exists, join, isfile, dirname, abspath, splitext


def make_dir(folder_name):
    if not exists(folder_name):
        makedirs(folder_name)


def get_hash(x: str):
    """Generate a hash from a string.
    """
    h = hashlib.md5(x.encode()) 
    return h.hexdigest()


class Cache(object):
    def __init__(self, func: Callable, cache_dir: str, cache_key: str):
        self.func = func
        self.cache_dir = join(cache_dir, cache_key)
        make_dir(self.cache_dir)
        self.cached_ids = [splitext(p)[0]for p in listdir(self.cache_dir)]

    def __call__(self, unique_id: str, *data):
        fpath = join(self.cache_dir, str('{}.npy'.format(unique_id)))

        if not exists(fpath):
            output = self.func(*data)

            self._write(output, fpath)
            self.cached_ids.append(unique_id)
        else:
            output = self._read(fpath)

        return self._read(fpath)

    def _write(self, x, fpath):
        np.save(fpath, x)
        # tmp = np.load(fpath, allow_pickle=True)

    def _read(self, fpath):
        return np.load(fpath, allow_pickle=True).item()
