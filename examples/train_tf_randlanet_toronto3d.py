import tensorflow as tf

# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets import Toronto3D, SemanticKITTI
from ml3d.tf.models import RandLANet

from ml3d.tf.dataloaders import TFDataloader
from ml3d.utils import Config
from os.path import abspath, dirname

# from tf2torch import load_tf_weights

config = 'ml3d/configs/randlanet_semantickitti.py'

cfg = Config.load_from_file(config)

dataset = SemanticKITTI(cfg.dataset)
model = RandLANet(cfg.model)

# pipeline = SemanticSegmentation(model, dataset, cfg.pipeline)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device     = torch.device('cpu')

# pipeline.run_train(device)

tf_data = TFDataloader(dataset = dataset.get_split('training'), model = model)# preprocess = model.preprocess, transform = model.transform, generator = model.get_batch_gen, cfg = model.cfg)

loader = tf_data.get_loader()
# print(loader)
for data in loader:
    print(len(data))