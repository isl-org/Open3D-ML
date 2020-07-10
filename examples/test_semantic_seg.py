import torch
# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets.semantickitti import ConfigSemanticKITTI, SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet


cfg     = ConfigSemanticKITTI
cfg.dataset_path = '/home/yiling/d2T/intel2020/datasets/semanticKITTI/data_odometry_velodyne/dataset/sequences_0.06'
dataset = SemanticKITTI(cfg)

model   = RandLANet(cfg)

pipeline = SemanticSegmentation(model, dataset, cfg)

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device  = torch.device('cpu')

pipeline.run_test(device)

