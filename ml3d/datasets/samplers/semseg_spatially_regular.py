import numpy as np
from tqdm import tqdm
import random

from ...utils import SAMPLER


class SemSegSpatiallyRegularSampler(object):
    """Spatially regularSampler sampler for semantic segmentation datasets."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)
        self.split = self.dataset.split

    def __len__(self):
        return self.length

    def initialize_with_dataloader(self, dataloader):
        self.min_possibilities = []
        self.possibilities = []

        self.length = len(dataloader)
        dataset = self.dataset

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

        def gen_train():
            for i in range(self.length):
                self.cloud_id = int(np.argmin(self.min_possibilities))
                yield self.cloud_id

        def gen_test():
            curr_could_id = 0
            while curr_could_id < self.length:
                if self.min_possibilities[curr_could_id] > 0.5:
                    curr_could_id = curr_could_id + 1
                    continue
                self.cloud_id = curr_could_id

                yield self.cloud_id

        if self.split in ['train', 'validation', 'valid', 'training']:
            gen = gen_train
        else:
            gen = gen_test
        return gen()

    def get_point_sampler(self):

        def _random_centered_gen(patchwise=True, **kwargs):
            if not patchwise:
                self.possibilities[self.cloud_id][:] = 1.
                self.min_possibilities[self.cloud_id] = 1.
                return
            pc = kwargs.get('pc', None)
            num_points = kwargs.get('num_points', None)
            radius = kwargs.get('radius', None)
            search_tree = kwargs.get('search_tree', None)
            if pc is None or num_points is None or (search_tree is None and
                                                    radius is None):
                raise KeyError(
                    "Please provide pc, num_points, and (search_tree or radius) \
                    for point_sampler in SemSegSpatiallyRegularSampler")

            cloud_id = self.cloud_id
            n = 0
            while n < 2:
                center_id = np.argmin(self.possibilities[cloud_id])
                center_point = pc[center_id, :].reshape(1, -1)

                if radius is not None:
                    idxs = search_tree.query_radius(center_point, r=radius)[0]
                elif num_points is not None:
                    if (pc.shape[0] < num_points):
                        diff = num_points - pc.shape[0]
                        idxs = np.array(range(pc.shape[0]))
                        idxs = list(idxs) + list(random.choices(idxs, k=diff))
                        idxs = np.asarray(idxs)
                    else:
                        idxs = search_tree.query(center_point,
                                                 k=num_points)[1][0]
                n = len(idxs)
                if n < 2:
                    self.possibilities[cloud_id][center_id] += 0.001

            random.shuffle(idxs)
            pc = pc[idxs]
            dists = np.sum(np.square((pc - center_point).astype(np.float32)),
                           axis=1)
            delta = np.square(1 - dists / np.max(dists))
            self.possibilities[cloud_id][idxs] += delta
            new_min = float(np.min(self.possibilities[cloud_id]))
            self.min_possibilities[cloud_id] = new_min

            return pc, idxs, center_point

        return _random_centered_gen


SAMPLER._register_module(SemSegSpatiallyRegularSampler)
