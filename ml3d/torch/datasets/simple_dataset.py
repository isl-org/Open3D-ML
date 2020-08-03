# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from tqdm import tqdm

import torch
from torch.multiprocessing import Pool
from torch.utils.data import Dataset

from ml3d.torch.utils import dataset_helper

from collections import namedtuple


class SimpleDataset(Dataset):
    def __init__(self,
                 *args,
                 dataset=None,
                 preprocess=None,
                 transform=None,
                 no_progress: bool = False,
                 **kwargs):
        self.dataset = dataset
        if preprocess is not None:
            desc = 'preprocess'
            cache_dir = getattr(dataset.cfg, 'cache_dir')
            assert cache_dir is not None, 'cache directory is not given'

            self.cache_convert = dataset_helper.Cache(
                preprocess,
                cache_dir=cache_dir,
                cache_key=dataset_helper._get_hash(repr(preprocess)))

            uncached = [
                idx for idx in range(len(dataset))
                if dataset.get_attr(idx)['name'] not in
                self.cache_convert.cached_ids
            ]
            if len(uncached) > 0:
                for idx in tqdm(
                        range(len(dataset)), desc=desc, disable=no_progress):
                    attr = dataset.get_attr(idx)
                    data = dataset.get_data(idx)
                    name = attr['name']

                    self.cache_convert(name, data, attr)

        else:
            self.cache_convert = None

        self.transform = transform

    def __getitem__(self, index):
        """Returns the item at index idx. """
        dataset = self.dataset
        attr = dataset.get_attr(index)
        data = (dataset.get_data(index)
                if self.cache_convert is None else self.cache_convert(
                    attr['name']))
        

        if self.transform is not None:
            data = self.transform(data, attr)

        inputs = {'data': data, 'attr': attr}

        return inputs

    def __len__(self):
        return len(self.dataset)
