import torch
from torch.utils.data import Sampler

class TorchSamplerWrapper(Sampler):
    """docstring for TorchSamplerWrapper"""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        return self.sampler.__iter__()

    def __len__(self):
        return len(self.sampler)
        

def get_sampler(sampler):
    return TorchSamplerWrapper(sampler)