from ml3d.datasets import (SemanticKITTI, ParisLille3D, Semantic3D, 
                                S3DIS, Toronto3D)
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from ml3d.utils import Config, get_module

def demo_dataset():
    # read data from datasets

    # dataset = SemanticKITTI(dataset_path="../dataset/SemanticKITTI",
    #                         use_cahe=False)
    datasets = []
    dataset = ParisLille3D(dataset_path="../dataset/Paris_Lille3D",
                            use_cahe=False)
    print(dataset.label_to_names)

    # print names of all pointcould
    split = dataset.get_split('test')
    for i in range(len(split)):
        attr = split.get_attr(i)
        print(attr['name'])

    split = dataset.get_split('train')
    for i in range(len(split)):
        data = split.get_data(i)
        print(data['point'].shape)

def demo_dataset_read():
    # read data from datasets

    dataset = SemanticKITTI(dataset_path="../dataset/SemanticKITTI",
                            use_cahe=False)
    print(dataset.label_to_names)

    # print names of all pointcould
    all_split = dataset.get_split('train')
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


def demo_train():
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


def demo_inference():
    # Inference and test example

    Pipeline = get_module("pipeline", "SemanticSegmentation", "torch")
    Model = get_module("model", "RandLANet", "torch")
    Dataset = get_module("dataset", "SemanticKITTI")

    # Initialize using default configuration in 
    # "ml3d/configs/default_cfgs/randlanet.yml"
    RandLANet = Model(
        ckpt_path="../dataset/checkpoints/randlanet_semantickitti.pth")

    # Initialize by specifying config file path
    SemanticKITTI = Dataset(cfg="ml3d/configs/default_cfgs/semantickitti.yml",
                            use_cahe=False)


    pipeline = Pipeline(model=RandLANet, 
                        dataset=SemanticKITTI,
                        device="gpu")

    # inference
    # get data
    train_split = SemanticKITTI.get_split("train")
    data = train_split.get_data(0)
    # restore weights
    pipeline.load_ckpt(RandLANet.cfg.ckpt_path, False)
    # run inference
    results = pipeline.run_inference(data)
    print(results)

    # test
    pipeline.run_test()


if __name__ == '__main__':
    demo_dataset()
    # demo_dataset_read()
    # demo_inference()
    # demo_train()
