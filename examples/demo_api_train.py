import torch
import yaml

from ml3d.datasets import SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from ml3d.utils import Config

def demo1():
    # read data from datasets
    dataset = SemanticKITTI(dataset_path="../dataset/SemanticKITTI",
                            use_cahe=True)
    print(dataset.label_to_names)

    # print names of all pointcould
    all_split = dataset.get_split('all')
    for i in range(len(all_split)):
        attr = all_split.get_attr(i)
        print(attr['name'])

    print(dataset.cfg.validation_split)
    # change the validation split
    dataset.cfg.validation_split = ['01']
    validation_split = dataset.get_split('val')
    for i in range(len(validation_split)):
        data = validation_split.get_data(i)
        print(data['point'].shape)


def demo2():
    # Initialize the training by passing parameters
    dataset = SemanticKITTI(dataset_path="../dataset/SemanticKITTI",
                            use_cahe=True)

    model = RandLANet(
                ckpt_path="../dataset/checkpoints/randlanet_semantickitti.pth",
                d_in=3)

    pipeline = SemanticSegmentation(model=model, dataset=dataset,
                                    max_epoch=100,
                                    batch_size=4,
                                    device="gpu")
    pipeline.run_train()


def demo3():
    # Initialize the training by config file
    from ml3d.utils import get_module
    import ml3d.torch

    Pipeline = get_module("pipeline", "SemanticSegmentation", "torch")
    Model = get_module("model", "RandLANet", "torch")
    Dataset = get_module("dataset", "SemanticKITTI")

    # Initialize using default configuration in 
    # "ml3d/configs/default_cfgs/randlanet.yml"
    RandLANet = Model()

    # Initialize by specifying config file path
    SemanticKITTI = Dataset(cfg="ml3d/configs/default_cfgs/semantickitti.yml",
                            use_cahe=False)


    pipeline = Pipeline(model=RandLANet, 
                        dataset=SemanticKITTI,
                        device="gpu")

    # run test
    pipeline.run_test()


if __name__ == '__main__':
    demo1()
    demo2()
    demo3()
