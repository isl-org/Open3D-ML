#import yaml

# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets import SemanticKITTI
from ml3d.tf.pipelines import SemanticSegmentation 
from ml3d.tf.models import RandLANet
from ml3d.utils import Config

py_config = 'ml3d/configs/randlanet_semantickitti.py'
# py_config 	= 'ml3d/configs/kpconv_semantickitti.py'
cfg         = Config.load_from_file(py_config)

dataset    	= SemanticKITTI(cfg.dataset)

model       = RandLANet(cfg.model)

pipeline    = SemanticSegmentation(model, dataset, cfg.pipeline)

#device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device     = torch.device('cpu')

pipeline.run_train()