import torch
# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets.semantickitti import SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from ml3d.torch.utils import Config

config_file = 'ml3d/torch/configs/randlanet_semantickitti.py'
cfg         = Config.load_from_file(config_file)

dataset 	= SemanticKITTI(cfg.dataset)

model   	= RandLANet(cfg.model)

pipeline 	= SemanticSegmentation(model, dataset, cfg.pipeline)

device  	= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device  	= torch.device('cpu')

pipeline.run_test(device)

