import argparse
import copy
import os
import os.path as osp
import yaml
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('framework', help='deep learning framework: tf or torch')
    parser.add_argument('-c', '--cfg_file', help='path to the config file')
    parser.add_argument('-m', '--model', help='network model')
    parser.add_argument('-p', '--pipeline', help='pipeline', default='SemanticSegmentation')
    parser.add_argument('-d', '--dataset', help='dataset')
    parser.add_argument('--cfg_model', help='path to the model\'s config file')
    parser.add_argument('--cfg_pipeline', help='path to the pipeline\'s config file')
    parser.add_argument('--cfg_dataset', help='path to the dataset\'s config file')
    parser.add_argument('--dataset_path', help='path to the dataset')
    parser.add_argument('--device', help='device to run the pipeline', default='gpu')
    parser.add_argument('--split', help='train or test', default='train')
    parser.add_argument('--work_dir', help='the dir to save logs and models')

    args, unknown = parser.parse_known_args()

    parser_extra = argparse.ArgumentParser(description='Extra arguments')
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser_extra.add_argument(arg)
    args_extra = parser_extra.parse_args(unknown)

    print("regular arguments")
    print(yaml.dump(vars(args)))

    print("extra arguments")
    print(yaml.dump(vars(args_extra)))

    return args, vars(args_extra) 


import ml3d.datasets 
from ml3d.utils import Config, get_module, convert_framework_name

def main():
    args, extra_dict = parse_args()

    framework = convert_framework_name(args.framework)
    if framework is 'torch':
        import ml3d.torch
    else:
        import ml3d.tf

    if args.cfg_file is not None:
        cfg = Config.load_from_file(args.cfg_file)
        Pipeline = get_module("pipeline", cfg.pipeline.name, framework)
        Model = get_module("model", cfg.model.name, framework)
        Dataset = get_module("dataset", cfg.dataset.name)

        dataset = Dataset(cfg=cfg.dataset, 
                        dataset_path=args.dataset_path, 
                        **extra_dict)
        model = Model(cfg=cfg.model, 
                        **extra_dict)
        pipeline = Pipeline(model=model, 
                            dataset=dataset, 
                            cfg=cfg.pipeline,
                            device=args.device,
                            **extra_dict)
    else: 
        if (args.pipeline and args.model and args.dataset) is None:
            raise ValueError("please specify pipeline, model, and dataset " +
                            "if no cfg_file given")

        Pipeline = get_module("pipeline", args.pipeline, framework)
        Model = get_module("model", args.model, framework)
        Dataset = get_module("dataset", args.dataset)

        dataset = Dataset(cfg=args.cfg_dataset, 
                        dataset_path=args.dataset_path, 
                        **extra_dict)
        model = Model(cfg=args.cfg_model, 
                    **extra_dict)
        pipeline = Pipeline(model=model, 
                            dataset=dataset, 
                            cfg=args.cfg_dataset, 
                            device=args.device,
                            **extra_dict)

    if args.split is 'train':
        pipeline.run_train()
    else:
        pipeline.run_test()

    
if __name__ == '__main__':
    main()
