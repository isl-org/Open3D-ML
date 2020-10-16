import argparse
import copy
import os
import sys
import os.path as osp
from pathlib import Path
import yaml
import time
import pprint


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('framework',
                        help='deep learning framework: tf or torch')
    parser.add_argument('-c', '--cfg_file', help='path to the config file')
    parser.add_argument('-m', '--model', help='network model')
    parser.add_argument('-p',
                        '--pipeline',
                        help='pipeline',
                        default='SemanticSegmentation')
    parser.add_argument('-d', '--dataset', help='dataset')
    parser.add_argument('--cfg_model', help='path to the model\'s config file')
    parser.add_argument('--cfg_pipeline',
                        help='path to the pipeline\'s config file')
    parser.add_argument('--cfg_dataset',
                        help='path to the dataset\'s config file')
    parser.add_argument('--dataset_path', help='path to the dataset')
    parser.add_argument('--device',
                        help='device to run the pipeline',
                        default='gpu')
    parser.add_argument('--split', help='train or test', default='train')
    parser.add_argument('--main_log_dir',
                        help='the dir to save logs and models')

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


import open3d.ml as _ml3d


def main():
    cmd_line = ' '.join(sys.argv[:])
    args, extra_dict = parse_args()

    framework = _ml3d.utils.convert_framework_name(args.framework)
    if framework == 'torch':
        import open3d.ml.torch as ml3d
    else:
        import tensorflow as tf
        import open3d.ml.tf as ml3d

        device = args.device
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if device == 'cpu':
                    tf.config.set_visible_devices([], 'GPU')
                elif device == 'gpu':
                    tf.config.set_visible_devices(gpus[0], 'GPU')
                else:
                    idx = device.split(':')[1]
                    tf.config.set_visible_devices(gpus[int(idx)], 'GPU')
            except RuntimeError as e:
                print(e)

    if args.cfg_file is not None:
        cfg = _ml3d.utils.Config.load_from_file(args.cfg_file)

        Pipeline = _ml3d.utils.get_module("pipeline", cfg.pipeline.name,
                                          framework)
        Model = _ml3d.utils.get_module("model", cfg.model.name, framework)
        Dataset = _ml3d.utils.get_module("dataset", cfg.dataset.name)

        cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
                        _ml3d.utils.Config.merge_cfg_file(cfg, args, extra_dict)

        dataset = Dataset(cfg_dict_dataset.pop('dataset_path', None),
                          **cfg_dict_dataset)
        model = Model(**cfg_dict_model)
        pipeline = Pipeline(model, dataset, **cfg_dict_pipeline)
    else:
        if (args.pipeline and args.model and args.dataset) is None:
            raise ValueError("please specify pipeline, model, and dataset " +
                             "if no cfg_file given")

        Pipeline = _ml3d.utils.get_module("pipeline", args.pipeline, framework)
        Model = _ml3d.utils.get_module("model", args.model, framework)
        Dataset = _ml3d.utils.get_module("dataset", args.dataset)


        cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
                        _ml3d.utils.Config.merge_module_cfg_file(args, extra_dict)

        dataset = Dataset(**cfg_dict_dataset)
        model = Model(**cfg_dict_model)
        pipeline = Pipeline(model, dataset, **cfg_dict_pipeline)

    with open(Path(__file__).parent / 'README.md', 'r') as f:
        readme = f.read()
    pipeline.cfg_tb = {
        'readme': readme,
        'cmd_line': cmd_line,
        'dataset': pprint.pformat(cfg_dict_dataset, indent=2),
        'model': pprint.pformat(cfg_dict_model, indent=2),
        'pipeline': pprint.pformat(cfg_dict_pipeline, indent=2)
    }

    if args.split == 'test':
        pipeline.run_test()
    else:
        pipeline.run_train()


if __name__ == '__main__':
    main()
