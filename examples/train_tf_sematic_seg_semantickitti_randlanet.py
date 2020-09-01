#import yaml

# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets import SemanticKITTI
from ml3d.tf.pipelines import SemanticSegmentation 
from ml3d.tf.models import RandLANet
from ml3d.utils import Config
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

py_config = 'ml3d/configs/randlanet_semantickitti.py'
# py_config 	= 'ml3d/configs/kpconv_semantickitti.py'
cfg         = Config.load_from_file(py_config)

dataset    	= SemanticKITTI(cfg.dataset)

model       = RandLANet(cfg.model)

pipeline    = SemanticSegmentation(model, dataset, cfg.pipeline)

#device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device     = torch.device('cpu')

pipeline.run_train()