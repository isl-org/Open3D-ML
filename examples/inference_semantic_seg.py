import torch
import yaml

# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets import SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet, KPFCNN
from ml3d.utils import Config

# from tf2torch import load_tf_weights

# yaml_config = 'ml3d/configs/randlanet_semantickitti.yaml'
py_config = 'ml3d/configs/randlanet_semantickitti.py'
# py_config = 'ml3d/configs/kpconv_semantickitti.py'
# py_config 	= 'ml3d/configs/kpconv_semantickitti.py'

cfg         = Config.load_from_file(py_config)

dataset    	= SemanticKITTI(cfg.dataset)
#dataset     = S3DIS(cfg.dataset)
datset_split = dataset.get_split('training')
data 		= datset_split.get_data(0)

model       = RandLANet(cfg.model)
# model       = KPFCNN(cfg.model)

pipeline    = SemanticSegmentation(model, dataset, cfg.pipeline)
pipeline.load_ckpt(model.cfg.ckpt_path, False)

device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device     = torch.device('cpu')

pred = pipeline.run_inference(data, device)

import numpy as np
np.set_printoptions(threshold=np.inf)
print(pred)
print(datset_split.get_attr(0)) 
gt = np.squeeze(model.inference_data['label'])
print(gt)
mask = gt == pred+1
print(gt.shape)
print(pred.shape)
print(np.sum(mask))
print(mask.shape)

print(np.sum(mask)/mask.shape[0])
# pipeline.run_train(device)