import torch
import yaml

# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets import SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet, KPFCNN
from ml3d.torch.utils import Config

# from tf2torch import load_tf_weights

# yaml_config = 'ml3d/torch/configs/randlanet_semantickitti.yaml'
# py_config = 'ml3d/torch/configs/randlanet_semantickitti.py'
py_config = 'ml3d/torch/configs/kpconv_semantickitti.py'
# py_config 	= 'ml3d/torch/configs/kpconv_semantickitti.py'
cfg         = Config.load_from_file(py_config)

dataset    	= SemanticKITTI(cfg.dataset)
#dataset     = S3DIS(cfg.dataset)

#model       = RandLANet(cfg.model)
model       = KPFCNN(cfg.model)

pipeline    = SemanticSegmentation(model, dataset, cfg.pipeline)

device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device     = torch.device('cpu')

pipeline.run_train(device)