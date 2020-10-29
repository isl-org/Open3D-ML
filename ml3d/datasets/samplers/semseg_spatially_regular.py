import numpy as np
from tqdm import tqdm
import random

from ...utils import SAMPLER

class SemSegSpatiallyRegularSampler(object):
    """Spatially regularSampler sampler for semantic segmentation datsets"""
    def __init__(self, dataset_split):
        self.dataset_split = dataset_split
        self.length = len(dataset_split)

    def __len__(self):
        return self.length

    def initialize_with_dataloader(self, dataloader):

        self.min_possibilities = []
        self.possibilities = []

        self.length = len(dataloader)
        dataset = self.dataset_split

        for step, inputs in enumerate(tqdm(train_loader, desc='training')):

        for index in range(len(dataset)):

            attr = dataset.get_attr(index)
            if dataloader.cache_convert:
                data = dataloader.cache_convert(attr['name'])
            elif dataloader.preprocess:
                data = dataloader.preprocess(dataset.get_data(index), attr)
            else:
                data = dataset.get_data(index)

            pc = data['point']
            self.possibilities += [np.random.rand(pc.shape[0]) * 1e-3]
            self.min_possibilities += [float(np.min(self.possibilities[-1]))]

    def get_cloud_sampler(self):
        def gen():
            for i in range(len(self.length)):
                self.cloud_id = int(np.argmin(self.min_possibilities))
                yield self.cloud_id
        return gen

    def get_point_sampler(self):
        def _random_centered_gen(**kwargs):
            pc = kwargs.get('pc', None)
            num_points = kwargs.get('num_points', None)
            search_tree = kwargs.get('search_tree', None)
            if pc is None or num_points is None or search_tree is None:
                raise KeyError(
                    "Please provide pc, num_points, and search_tree \
                    for point_sampler in SemSegSpatiallyRegularSampler")

            cloud_id = self.cloud_id
            center_id = np.argmin(self.possibilities[cloud_id])
            center_point = pc[center_id, :].reshape(1, -1)

            if (pc.shape[0] < num_points):
                idxs = np.array(range(pc.shape[0]))
                idxs = list(idxs) + list(random.choices(idxs, k=diff))
            else:
                idxs = search_tree.query(center_point, k=num_points)[1][0]
            random.shuffle(idxs)
            pc = pc[idxs]

            dists = np.sum(np.square((pc-center_point).astype(np.float32)), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            self.possibilities[cloud_id] += delta
            new_min = float(np.min(self.possibilities[cloud_id]))
            self.min_possibilities[cloud_id] = new_min

            return pc, idxs, center_point

        return _random_centered_gen



SAMPLER._register_module(SemSegSpatiallyRegularSampler)