"""File with tensorflow dataset code that uses ml3d/datasets/scannet.py"""
from ...datasets.scannet import ScanNet as _ScanNet

class ScanNet:

    def __init__(self,root):
        self.scannet = _ScanNet(root)

    def __getitem__(self, index):
        data = self.scannet[index]
        data['feats'] = data['colors'].astype('float32')
        return data

    def __len__(self):
        return len(self.scannet)
    
