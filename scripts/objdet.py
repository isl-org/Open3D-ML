import torch
import numpy as np
import open3d.ml as ml3d

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


test_split = TorchDataloader(dataset=dataset.get_split('training'),
                                preprocess=model.preprocess,
                                transform=model.transform,
                                use_cache=False,#dataset.cfg.use_cache,
                                shuffle=False)
data = test_split[0]['data']

# run inference on a single example.
result = pipeline.run_inference(data)

boxes = []
for label in data['label']:
    front = np.array((0, 0, 1))
    up = np.array((0, 1, 0))
    left = np.array((1, 0, 0))

    rot = np.array(((1.0, 0.0, 1,0),
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 1,0)))

    box = BoundingBox3D(label.loc, front, up, left, (label.w, label.l, label.h),
                    label.cls_id, 0.0)
    boxes.append(box)

vis = Visualizer()
vis.visualize([{
        "name": "KITTI",
        'points': data['point'][:,:3]
    }], bounding_boxes=boxes)



"""
import math
import numpy as np
import open3d as o3d
from open3d.ml.datasets import ParisLille3D
from open3d.ml.vis import Visualizer, LabelLUT, BoundingBox3D
import random
import sys

from open3d.visualization import gui

def make_random_coord_system():
    yaw = random.random() * 2.0 * math.pi
    pitch = random.random() * 2.0 * math.pi
    roll = random.random() * 2.0 * math.pi
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    rot = np.array(((cos_yaw*cos_pitch, cos_yaw*sin_pitch*sin_roll - sin_yaw*cos_roll, cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll),
                    (sin_yaw*cos_pitch, sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll, sin_yaw*sin_pitch*cos_roll - cos_yaw*sin_roll),
                    (-sin_pitch, cos_pitch*sin_roll, cos_pitch*cos_roll)))
    front = rot.dot(np.array((0, 0, 1)))
    up = rot.dot(np.array((0, 1, 0)))
    left = rot.dot(np.array((1, 0, 0)))
    return (front, up, left)

class MockSplit:
    def __init__(self, dataset, name, downsample):
        self._downsample = downsample
        self.dataset = dataset
        self._split = self.dataset.get_split(name)
        self.path_list = self._split.path_list

    def __len__(self):
        return len(self._split)

    def get_data(self, idx):
        data = self._split.get_data(idx)
        labels = data["label"]
        points = data["point"]
        box_skip = 200000

        if self._downsample:
            box_skip = 50
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            bounds = cloud.get_axis_aligned_bounding_box()
            size = min(bounds.max_bound[0] - bounds.min_bound[0],
                       bounds.max_bound[1] - bounds.min_bound[1],
                       bounds.max_bound[2] - bounds.min_bound[2]) / 20.0
            _, _, indexList = cloud.voxel_down_sample_and_trace(size,
                                                                bounds.min_bound,
                                                                bounds.max_bound,
                                                                False)
            indices = []
            for il in indexList:
                indices.append(il[0])
            labels = labels[indices]
            points = points[indices]
            data["label"] = labels
            data["point"] = points

        boxes = []
        for i in range(0, len(points), box_skip):
            front, up, left = make_random_coord_system()
            box = BoundingBox3D(points[i], front, up, left, (0.5, 0.5, 1.0),
                                labels[i], 0.0)
            boxes.append(box)

        data['bounding_boxes'] = boxes  # this is what the visualizer looks for
        return data

    def get_attr(self, idx):
        return self._split.get_attr(idx)

class MockDataset:
    def __init__(self, path, downsample=False):
        self._downsample = downsample
        self._dataset = ParisLille3D(path)
        self.name = self._dataset.name
        self.label_to_names = self._dataset.label_to_names
        self._split = self._dataset.get_split("all")

    def get_split(self, name):
        return MockSplit(self._dataset, name, self._downsample)

def main():
    which = sys.argv[1]
    path = sys.argv[2]
    dataset = MockDataset(path, downsample=(which == "dataset"))
    v = Visualizer()

    if which == "dataset":
        v.visualize_dataset(dataset, "all", [0, 1])
    else:
        lut = LabelLUT()
        for val in sorted(dataset.label_to_names.keys()):
            lut.add_label(dataset.label_to_names[val], val)

        src = dataset.get_split("all").get_data(0)
        data = { 'name': "Lille1_1",
                 'points': src['point'],
                 'labels': src['label'],
                 'feat': src['feat'] }
        v.visualize([data], lut, bounding_boxes=src["bounding_boxes"])
        

if __name__ == "__main__":
    main()

"""