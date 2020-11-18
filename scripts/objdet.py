import torch

import open3d.ml as ml3d

from ml3d.vis import Visualizer
from ml3d.datasets import KITTI

from ml3d.torch.pipelines import ObjectDetection
from ml3d.torch.models import PointPillars


dataset_path = "/home/prantl/obj_det/mmdetection3d/data/kitti"
ckpt_path = "/home/prantl/obj_det/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth"


model = PointPillars()
dataset = KITTI(dataset_path)
pipeline = ObjectDetection(model, dataset, device="gpu")
    
# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)

test_split = dataset.get_split("training")
data = test_split.get_data(0)

# run inference on a single example.
result = pipeline.run_inference(data)

vis = Visualizer()
vis.visualize(data['point'][:,:3])
