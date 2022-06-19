import torch
from torch import nn
import open3d.ml.torch as ml3d
from .base_model_objdet import BaseModel

class PVRCNN_plus_plus(BaseModel):
    def __init__(self, 
                 name = "PVRCNN++",
                 device="cuda",
                 point_cloud_range=[0, -40.0, -3, 70.0, 40.0, 1],
                 classes=['car'],
                 **kwargs):
        super().__init__(name=name,
                         point_cloud_range=point_cloud_range,
                         device=device,
                         **kwargs)
        self.point_cloud_range = point_cloud_range
        self.classes = classes
        self.device = device
        self.to(device)

    @torch.no_grad()
    def voxelize(self, points):
        """Apply voxelization to points."""
        pass
    
    def get_optimizer(self, cfg):
        pass

    def get_loss(self, results, inputs):
        pass

    def preprocess(self, data, attr):
        pass

    def transform(self, data, attr):
        pass

    def inference_end(self, results, inputs):
        pass

"""
Voxelization Layer
Mean in OpenPCDet implementation
"""
class PVRCNNPlusPlusVoxelization(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class PVRCNNPlusPlusBackbone3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class PVRCNNPlusPlusBEVModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class RPNModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class PVRCNNPlusPlusVectorPoolAggregationModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class PVRCNNPlusPlusVoxelSetAbstraction(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def get_keypoints(self, boxes, points):
        pass


class 


        