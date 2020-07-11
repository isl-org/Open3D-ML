import torch
import numpy as np
# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets.semantickitti import SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from ml3d.torch.utils import Config


import open3d as o3d

config_file = 'ml3d/torch/configs/randlanet_semantickitti.py'
cfg         = Config.load_from_file(config_file)
dataset 	= SemanticKITTI(cfg.dataset)

model   	= RandLANet(cfg.model)

pipeline 	= SemanticSegmentation(model, dataset, cfg.pipeline)

device  	= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device  = torch.device('cpu')

pc_file 	= 'data/demo/fragment.ply'
pcd 		= o3d.io.read_point_cloud(pc_file)
points 		= np.asarray(pcd.points).astype(np.float32)
print(points.shape)
# input:  points: nv x 3   numpy
# output: result: n        tensor    
result 		= pipeline.run_inference(points, device)
print(result.size())