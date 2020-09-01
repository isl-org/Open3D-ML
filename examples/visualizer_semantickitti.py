import torch
import yaml

# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets import SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet, KPFCNN
from ml3d.utils import Config

py_config = 'ml3d/configs/randlanet_semantickitti.py'

cfg         = Config.load_from_file(py_config)

dataset    	= SemanticKITTI(cfg.dataset)

dataset_split = dataset.get_split('all')
for i in range(len(dataset_split)):
	# data = dataset_split.get_data(i)
	attr = dataset_split.get_attr(i)
	print(attr['name'])
print(len(dataset_split))