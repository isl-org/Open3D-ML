from ml3d.datasets import (SemanticKITTI, ParisLille3D, Semantic3D, 
                                S3DIS, Toronto3D)
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from ml3d.utils import Config, get_module

def demo_read_data():
    # read data from datasets
    dataset = SemanticKITTI(dataset_path="../dataset/SemanticKITTI",
                            use_cache=False)
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
                            use_cache=True)

    model = RandLANet(
                ckpt_path="../dataset/checkpoints/randlanet_semantickitti.pth",
                dim_input=3)

    pipeline = SemanticSegmentation(model=model, dataset=dataset,
                                    max_epoch=100,
                                    batch_size=4,
                                    device="gpu")

    pipeline.run_train()


def demo_inference():
    # Inference and test example
    from ml3d.tf.pipelines import SemanticSegmentation 
    from ml3d.tf.models import RandLANet

    Pipeline = get_module("pipeline", "SemanticSegmentation", "tf")
    Model = get_module("model", "RandLANet", "tf")
    Dataset = get_module("dataset", "SemanticKITTI")

    # Initialize using default configuration in 
    # "ml3d/configs/default_cfgs/randlanet.yml"
    RandLANet = Model(
        ckpt_path="../dataset/checkpoints/randlanet_semantickitti.pth")

    # Initialize by specifying config file path
    cfg = Config.load_from_file("ml3d/configs/default_cfgs/semantickitti.yml")
    cfg.use_cache=False
    SemanticKITTI = Dataset(**cfg)


    pipeline = Pipeline(model=RandLANet, 
                        dataset=SemanticKITTI)

    # inference
    # get data
    train_split = SemanticKITTI.get_split("train")
    data = train_split.get_data(0)
    # restore weights
    # pipeline.load_ckpt(RandLANet.cfg.ckpt_path, False)
    # run inference
    results = pipeline.run_inference(data)
    print(results)

    # test
    pipeline.run_test()


if __name__ == '__main__':
    demo_read_data()
    demo_inference()
    demo_train()
