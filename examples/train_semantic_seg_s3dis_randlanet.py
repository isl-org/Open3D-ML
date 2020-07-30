import torch

# there should be pipeline. pipeline is bigger that randlanet
from ml3d.torch.datasets import S3DIS
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from ml3d.torch.utils import Config

# from tf2torch import load_tf_weights

config = 'ml3d/torch/configs/randlanet_s3dis.py'
cfg         = Config.load_from_file(config)

dataset     = S3DIS(cfg.dataset)

model       = RandLANet(cfg.model)

pipeline    = SemanticSegmentation(model, dataset, cfg.pipeline)

device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device     = torch.device('cpu')

pipeline.run_train(device)