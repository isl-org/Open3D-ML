from open3d.ml.datasets import (SemanticKITTI, ParisLille3D, Semantic3D, S3DIS,
                                Toronto3D)
from open3d.ml.torch.pipelines import SemanticSegmentation
from open3d.ml.torch.models import RandLANet
from open3d.ml.utils import Config, get_module

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo for training and inference')
    parser.add_argument('--path_semantickitti',
                        help='path to semantiSemanticKITTI',
                        required=True)
    parser.add_argument('--path_ckpt_randlanet',
                        help='path to RandLANet checkpoint')

    args, _ = parser.parse_known_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args


def demo_train(args):
    # Initialize the training by passing parameters
    dataset = SemanticKITTI(args.path_semantickitti, use_cache=True)

    model = RandLANet(dim_input=3)

    pipeline = SemanticSegmentation(model=model, dataset=dataset, max_epoch=100)

    pipeline.run_train()


def demo_inference(args):
    # Inference and test example
    from open3d.ml.tf.pipelines import SemanticSegmentation
    from open3d.ml.tf.models import RandLANet

    Pipeline = get_module("pipeline", "SemanticSegmentation", "tf")
    Model = get_module("model", "RandLANet", "tf")
    Dataset = get_module("dataset", "SemanticKITTI")

    RandLANet = Model(ckpt_path=args.path_ckpt_randlanet)

    # Initialize by specifying config file path
    SemanticKITTI = Dataset(args.path_semantickitti, use_cache=False)

    pipeline = Pipeline(model=RandLANet, dataset=SemanticKITTI)

    # inference
    # get data
    train_split = SemanticKITTI.get_split("train")
    data = train_split.get_data(0)
    # restore weights

    # run inference
    results = pipeline.run_inference(data)
    print(results)

    # test
    pipeline.run_test()


if __name__ == '__main__':
    args = parse_args()
    demo_train(args)
    demo_inference(args)
