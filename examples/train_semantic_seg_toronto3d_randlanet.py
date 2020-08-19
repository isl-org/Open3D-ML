import torch

# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets import Toronto3D
from ml3d.torch.pipelines import SemanticSegmentation
from ml3d.torch.models import RandLANet
from ml3d.utils import Config

# from tf2torch import load_tf_weights

config = 'ml3d/configs/randlanet_toronto3d.py'
cfg = Config.load_from_file(config)

dataset = Toronto3D(cfg.dataset)

model = RandLANet(
    d_in = 6,
    d_out = [16, 64, 128, 256, 512],
    d_feature = 8,
    num_classes = 8,
    num_layers = 5
)

pipeline = SemanticSegmentation(
    model,
    dataset,
    batch_size = 1,
    learning_rate = 1e-2
    )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device     = torch.device('cpu')

pipeline.run_train(device)