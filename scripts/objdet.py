
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

import torch

from tqdm import tqdm

Model = _ml3d.utils.get_module("model", "PointPillars", "torch")
Dataset = _ml3d.utils.get_module("dataset", "KITTI")
Pipeline = _ml3d.utils.get_module("pipeline", "ObjectDetection",
                                        "torch")

def main():
    dataset = Dataset("/home/prantl/obj_det/mmdetection3d/data/kitti")
    model = Model()
    pipeline = Pipeline(model, dataset)

    pipeline.run_train()
    pipeline.run_test()

if __name__ == '__main__':
    main()
