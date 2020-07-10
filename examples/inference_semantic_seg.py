import torch
import numpy as np
# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets.semantickitti import ConfigSemanticKITTI, SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet


import open3d as o3d

cfg     = ConfigSemanticKITTI
dataset = SemanticKITTI(cfg)

model   = RandLANet(cfg)

pipeline = SemanticSegmentation(model, dataset, cfg)

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device  = torch.device('cpu')

pointcloud_file = 'data/demo/fragment.ply'
pcd = o3d.io.read_point_cloud(pointcloud_file)
points = np.asarray(pcd.points).astype(np.float32)
print(points.shape)
result = pipeline.run_inference(points, device)
print(result.shape)