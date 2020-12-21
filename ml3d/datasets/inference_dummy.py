import numpy as np
import os, sys, glob, pickle
import logging

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET, get_module

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
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
