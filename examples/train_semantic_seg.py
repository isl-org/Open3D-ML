import torch
import yaml

# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets.semantickitti import SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from ml3d.torch.utils import Config


#py_config 	= 'ml3d/torch/configs/randlanet_semantickitti.py'
yaml_config = 'ml3d/torch/configs/randlanet_semantickitti.yaml'
cfg         = Config.load_from_file(yaml_config)

dataset     = SemanticKITTI(cfg.dataset)

model       = RandLANet(cfg.model)

pipeline    = SemanticSegmentation(model, dataset, cfg.pipeline)

device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device     = torch.device('cpu')

pipeline.run_train(device)