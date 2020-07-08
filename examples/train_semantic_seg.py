import torch
# there should be pipeline. pipeline is bigger that randlanet
from ml3d.datasets.semantickitti import ConfigSemanticKITTI, SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from tf2torch import load_tf_weights


cfg     = ConfigSemanticKITTI
cfg.dataset_path = '/home/yiling/d2T/intel2020/datasets/semanticKITTI/data_odometry_velodyne/dataset/sequences_0.06'
dataset = SemanticKITTI(cfg)



gpt2_checkpoint_path = '/home/yiling/d2T/intel2020/RandLA-Net/models/SemanticKITTI/snap-277357'
model   = RandLANet(cfg)
load_tf_weights(model, gpt2_checkpoint_path)


pipeline = SemanticSegmentation(model, dataset, cfg)

device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device  = torch.device('cpu')

#pipeline.run_test(device)
pipeline.run_train(device)

