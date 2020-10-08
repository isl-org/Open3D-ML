from abc import abstractmethod
from tqdm import tqdm
import torch
from torch.multiprocessing import Pool
from torch.utils.data import Dataset
from collections import namedtuple

from ...utils import Cache, get_hash


class TorchDataloader(Dataset):
    """
    Data loader for torch framework.
    """

    def __init__(self,
                 dataset=None,
                 preprocess=None,
                 transform=None,
                 use_cache=True,
                 steps_per_epoch=None,
                 **kwargs):
        """
        Initialize

        Args:
            dataset: ml3d dataset class.
            dataset: model's preprocess method.
            devce: model's transform mthod.
            use_cache: whether to use cached preprocessed data.
            steps_per_epch: steps per epoch. The step number will be the 
                number of samples in the data if steps_per_epoch=None
            kwargs:
        Returns:
            class: The corresponding class.
        """
        self.dataset = dataset
        self.preprocess = preprocess
        self.steps_per_epoch = steps_per_epoch

        if preprocess is not None and use_cache:
            cache_dir = getattr(dataset.cfg, 'cache_dir')
            assert cache_dir is not None, 'cache directory is not given'

            self.cache_convert = Cache(preprocess,
                                       cache_dir=cache_dir,
                                       cache_key=get_hash(repr(preprocess)))

            uncached = [
                idx for idx in range(len(dataset)) if dataset.get_attr(idx)
                ['name'] not in self.cache_convert.cached_ids
            ]
            if len(uncached) > 0:
                for idx in tqdm(range(len(dataset)), desc='preprocess'):
                    attr = dataset.get_attr(idx)
                    name = attr['name']
                    if name in self.cache_convert.cached_ids:
                        continue
                    data = dataset.get_data(idx)
                    # cache the data
                    self.cache_convert(name, data, attr)

        else:
            self.cache_convert = None

        self.transform = transform

    def __getitem__(self, index):
        """Returns the item at index idx. """
        dataset = self.dataset
        index = index % len(dataset)

        attr = dataset.get_attr(index)
        if self.cache_convert:
            data = self.cache_convert(attr['name'])
        elif self.preprocess:
            data = self.preprocess(dataset.get_data(index), attr)
        else:
            data = dataset.get_data(index)

        if self.transform is not None:
            data = self.transform(data, attr)

        inputs = {'data': data, 'attr': attr}

        return inputs

    def __len__(self):
        """Returns the number of steps for one epoch"""
        if self.steps_per_epoch is not None:
            steps_per_epoch = self.steps_per_epoch
        else:
            steps_per_epoch = len(self.dataset)
        return steps_per_epoch
