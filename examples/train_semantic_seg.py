from ml3d.datasets import Toronto3D
from ml3d.tf.pipelines import SemanticSegmentation 
from ml3d.tf.models import RandLANet, KPFCNN
from ml3d.utils import Config
from ml3d.tf.dataloaders import TFDataloader

py_config = 'ml3d/configs/kpconv_toronto3d.yml'

cfg         = Config.load_from_file(py_config)

dataset    	= Toronto3D(cfg.dataset)
model       = KPFCNN(cfg.model)

pipeline    = SemanticSegmentation(model, dataset, cfg.pipeline)

pipeline.run_train()
# import numpy as np
# np.set_printoptions(threshold=np.inf)

# pred = results['predict_labels']

# mask = gt == pred+1
# print(gt.shape)
# print(pred.shape)
# print(np.sum(mask))
# print(mask.shape)

# print(np.sum(mask)/mask.shape[0])
