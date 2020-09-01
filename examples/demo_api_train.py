import torch
import yaml

from ml3d.datasets import SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from ml3d.utils import Config


def demo1():
    dataset = SemanticKITTI(dataset_path="../dataset/SemanticKITTI",
                            use_cahe=True)
    
    model = RandLANet(
                ckpt_path="../dataset/checkpoints/randlanet_semantickitti.pth",
                d_in=3)

    pipeline = SemanticSegmentation(model=model, dataset=dataset,
                                    max_epoch=100,
                                    batch_size=4,
                                    device="gpu")
    pipeline.run_train(device)

if __name__ == '__main__':
    demo1()
