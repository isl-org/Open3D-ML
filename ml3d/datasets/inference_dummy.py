import logging

from .base_dataset import BaseDatasetSplit
from ..utils import DATASET, get_module

log = logging.getLogger(__name__)


class InferenceDummySplit(BaseDatasetSplit):

    def __init__(self, inference_data):
        self.split = 'test'
        self.inference_data = inference_data
        self.cfg = {}
        sampler_cls = get_module('sampler', 'SemSegSpatiallyRegularSampler')
        self.sampler = sampler_cls(self)

    def __len__(self):
        return 1

    def get_data(self, idx):
        return self.inference_data

    def get_attr(self, idx):
        pc_path = 'inference_data'
        split = self.split
        attr = {'idx': 0, 'name': 'inference', 'path': pc_path, 'split': split}
        return attr


DATASET._register_module(InferenceDummySplit)
