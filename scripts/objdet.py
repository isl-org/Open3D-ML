import torch
import numpy as np
import open3d.ml as ml3d
import math

from ml3d.vis import Visualizer, BoundingBox3D
from ml3d.datasets import KITTI

from ml3d.torch.pipelines import ObjectDetection
from ml3d.torch.models import PointPillars
from ml3d.torch.dataloaders import TorchDataloader


dataset_path = "/Users/lprantl/Open3D-ML/data/kitti"
ckpt_path = "/Users/lprantl/Open3D-ML/checkpoints/pointpillars_kitti_3class.pth"


model = PointPillars()
dataset = KITTI(dataset_path)
pipeline = ObjectDetection(model, dataset, device="gpu")
    
# load the parameters.
pipeline.load_ckpt(ckpt_path=ckpt_path)


#test_split = TorchDataloader(dataset=dataset.get_split('training'),
#                                preprocess=model.preprocess,
#                                transform=model.transform,
#                                use_cache=False,#dataset.cfg.use_cache,
#                                shuffle=False)

pc_path = "/Users/lprantl/Open3D-ML/data/kitti/training/velodyne/000001.bin"
label_path = pc_path.replace('velodyne',
                                'label_2').replace('.bin', '.txt')
calib_path = label_path.replace('label_2', 'calib')

pc = dataset.read_lidar(pc_path)
calib = dataset.read_calib(calib_path)
label = dataset.read_label(label_path, calib)

data = {
    'point': pc,
    'feat': None,
    'calib': calib,
    'bounding_boxes': label,
}

data = model.transform(data, None)

# run inference on a single example.
result = pipeline.run_inference(data)

boxes = data['bounding_boxes']
boxes.extend(result)

vis = Visualizer()
vis.visualize([{
        "name": "KITTI",
        'points': data['point'][:,:3]
    }], bounding_boxes=boxes)

