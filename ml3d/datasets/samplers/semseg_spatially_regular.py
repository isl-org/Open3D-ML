import numpy as np
import random

from ...utils import SAMPLER

class SemSegSpatiallyRegularSampler(object):
    """Spatially regularSampler sampler for semantic segmentation datsets"""
    def __init__(self, dataset_split):
        self.dataset_split = dataset_split

    def get_cloud_sampler(self):
        class _RandomSampler():
            def __init__(self, num_samples):
                self.num_samples = num_samples
            
            def __iter__(self):
                def gen():
                    ids = np.random.permutation(self.num_samples)
                    for i in ids:
                        yield i

                return gen()

            def __len__(self):
                return self.num_samples

        return _RandomSampler(len(self.dataset_split))

    def get_point_sampler(self):
        def _random_centered_gen(**kwargs):
            pc = kwargs.get('pc', None)
            num_points = kwargs.get('num_points', None)
            search_tree = kwargs.get('search_tree', None)
            if pc is None or num_points is None or search_tree is None:
                raise KeyError(
                    "Please provide pc, num_points, and search_tree \
                    for point_sampler in SemSegSpatiallyRegularSampler")

            center_idx = np.random.choice(len(pc), 1)
            center_point = pc[center_idx, :].reshape(1, -1)

            if (pc.shape[0] < num_points):
                idxs = np.array(range(pc.shape[0]))
                idxs = list(idxs) + list(random.choices(idxs, k=diff))
            else:
                idxs = search_tree.query(center_point, k=num_points)[1][0]
            random.shuffle(idxs)
            pc = pc[idxs]
            return pc, idxs, center_point

        return _random_centered_gen



SAMPLER._register_module(SemSegSpatiallyRegularSampler)