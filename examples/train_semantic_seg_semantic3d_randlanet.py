import torch

# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets import Semantic3D
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from ml3d.utils import Config

# from tf2torch import load_tf_weights

config = 'ml3d/configs/randlanet_semantic3d.py'
cfg         = Config.load_from_file(config)

dataset     = Semantic3D(cfg.dataset)

model       = RandLANet(cfg.model)

pipeline    = SemanticSegmentation(model, dataset, cfg.pipeline)

device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device     = torch.device('cpu')

pipeline.run_train(device)