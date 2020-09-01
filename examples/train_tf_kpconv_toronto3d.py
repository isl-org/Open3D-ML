from ml3d.datasets import Toronto3D
from ml3d.tf.models import KPFCNN
from ml3d.tf.pipelines import SemanticSegmentation 
from ml3d.utils import Config
from ml3d.configs

from os.path import abspath, dirname


config = dirname(abspath(__file__)) + '/../ml3d/configs/kpconv_toronto3d.py'
cfg = Config.load_from_file(config)

dataset = Toronto3D(cfg.dataset)

model = KPFCNN(cfg.model)

pipeline = SemanticSegmentation(model, dataset, cfg.pipeline)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device     = torch.device('cpu')

pipeline.run_train()