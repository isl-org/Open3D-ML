import hashlib
from typing import Callable
import numpy as np

from os import makedirs, listdir
from os.path import exists, join, splitext


def make_dir(folder_name):
    """Create a directory.

    If already exists, do nothing
    """
    if not exists(folder_name):
        makedirs(folder_name)


def get_hash(x: str):
    """Generate a hash from a string."""
    h = hashlib.md5(x.encode())
    return h.hexdigest()


class Cache(object):
    """Cache converter for preprocessed data."""

    def __init__(self, func: Callable, cache_dir: str, cache_key: str):
        """Initialize.

        Args:
            func: preprocess function of a model.
            cache_dir: directory to store the cache.
            cache_key: key of this cache
        Returns:
            class: The corresponding class.
        """
        self.func = func
        self.cache_dir = join(cache_dir, cache_key)
        make_dir(self.cache_dir)
        self.cached_ids = [splitext(p)[0] for p in listdir(self.cache_dir)]

    def __call__(self, unique_id: str, *data):
        """Call the converter. If the cache exists, load and return the cache,
        otherwise run the preprocess function and store the cache.

        Args:
            unique_id: A unique key of this data.
            data: Input to the preprocess function.

        Returns:
            class: Preprocessed (cache) data.
        """
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

    def _read(self, fpath):
        return np.load(fpath, allow_pickle=True).item()
