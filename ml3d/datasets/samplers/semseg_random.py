import numpy as np

class SemSegRandomSampler(object):
    """Random sampler for semantic segmentation datsets"""
    def __init__(self, dataset_split):
        self.dataset_split = dataset_split

    def get_cloud_sampler():
        class _RandomSampler():
            """Generator for _RandomSampler"""
            def __init__(self, num_samples):
                self.num_samples = num_samples
                
            def __iter__(self):
                return np.random.randint(self.num_samples)
                
            def __len__(self):
                return self.num_samples

        return _RandomSampler(len(self.dataset_split))

    def get_point_sampler():

Sampler._register_module(SemSegRandomSampler)