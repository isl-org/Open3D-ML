import numpy as np
import argparse
import sys
import yaml
import pprint
import os
import torch.distributed as dist
from torch import multiprocessing

from pathlib import Path


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
    parser.add_argument('--ckpt_path', help='path to the checkpoint')
    parser.add_argument('--device',
                        help='devices to run the pipeline',
                        default='cuda')
    parser.add_argument('--device_ids',
                        nargs='+',
                        help='cuda device list',
                        default=['0'])
    parser.add_argument('--split', help='train or test', default='train')
    parser.add_argument('--mode', help='additional mode', default=None)
    parser.add_argument('--max_epochs', help='number of epochs', default=None)
    parser.add_argument('--batch_size', help='batch size', default=None)
    parser.add_argument('--main_log_dir',
                        help='the dir to save logs and models')
    parser.add_argument('--seed', help='random seed', default=0)

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
    args.device, args.device_ids = _ml3d.utils.convert_device_name(
        args.device, args.device_ids)
    rng = np.random.default_rng(args.seed)
    if framework == 'torch':
        import open3d.ml.torch as ml3d
        import torch.multiprocessing as mp
        import torch.distributed as dist
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
                elif device == 'cuda':
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

        if args.mode is not None:
            cfg_dict_model["mode"] = args.mode
        if args.max_epochs is not None:
            cfg_dict_pipeline["max_epochs"] = args.max_epochs
        if args.batch_size is not None:
            cfg_dict_pipeline["batch_size"] = args.batch_size

        cfg_dict_dataset['seed'] = rng
        cfg_dict_model['seed'] = rng
        cfg_dict_pipeline['seed'] = rng

        cfg_dict_pipeline["device"] = args.device
        cfg_dict_pipeline["device_ids"] = args.device_ids

    else:
        if (args.pipeline and args.model and args.dataset) is None:
            raise ValueError("Please specify pipeline, model, and dataset " +
                             "if no cfg_file given")

        Pipeline = _ml3d.utils.get_module("pipeline", args.pipeline, framework)
        Model = _ml3d.utils.get_module("model", args.model, framework)
        Dataset = _ml3d.utils.get_module("dataset", args.dataset)


        cfg_dict_dataset, cfg_dict_pipeline, cfg_dict_model = \
                        _ml3d.utils.Config.merge_module_cfg_file(args, extra_dict)

        cfg_dict_dataset['seed'] = rng
        cfg_dict_model['seed'] = rng
        cfg_dict_pipeline['seed'] = rng

    with open(Path(__file__).parent / 'README.md', 'r') as f:
        readme = f.read()

    cfg_tb = {
        'readme': readme,
        'cmd_line': cmd_line,
        'dataset': pprint.pformat(cfg_dict_dataset, indent=2),
        'model': pprint.pformat(cfg_dict_model, indent=2),
        'pipeline': pprint.pformat(cfg_dict_pipeline, indent=2)
    }
    args.cfg_tb = cfg_tb
    args.distributed = framework == 'torch' and args.device != 'cpu' and len(
        args.device_ids) > 1

    if not args.distributed:
        dataset = Dataset(**cfg_dict_dataset)
        model = Model(**cfg_dict_model, mode=args.mode)
        pipeline = Pipeline(model, dataset, **cfg_dict_pipeline)

        pipeline.cfg_tb = cfg_tb

        if args.split == 'test':
            pipeline.run_test()
        else:
            pipeline.run_train()

    else:
        mp.spawn(main_worker,
                 args=(Dataset, Model, Pipeline, cfg_dict_dataset,
                       cfg_dict_model, cfg_dict_pipeline, args),
                 nprocs=len(args.device_ids))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, Dataset, Model, Pipeline, cfg_dict_dataset,
                cfg_dict_model, cfg_dict_pipeline, args):
    world_size = len(args.device_ids)
    setup(rank, world_size)

    cfg_dict_dataset['rank'] = rank
    cfg_dict_model['rank'] = rank
    cfg_dict_pipeline['rank'] = rank

    rng = np.random.default_rng(args.seed + rank)
    cfg_dict_dataset['seed'] = rng
    cfg_dict_model['seed'] = rng
    cfg_dict_pipeline['seed'] = rng

    device = f"cuda:{args.device_ids[rank]}"
    print(f"rank = {rank}, world_size = {world_size}, gpu = {device}")

    cfg_dict_model['device'] = device
    cfg_dict_pipeline['device'] = device

    dataset = Dataset(**cfg_dict_dataset)
    model = Model(**cfg_dict_model, mode=args.mode)
    pipeline = Pipeline(model,
                        dataset,
                        distributed=args.distributed,
                        **cfg_dict_pipeline)

    with open(Path(__file__).parent / 'README.md', 'r') as f:
        readme = f.read()
    pipeline.cfg_tb = args.cfg_tb

    if args.split == 'test':
        if rank == 0:
            pipeline.run_test()
    else:
        pipeline.run_train()

    cleanup()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    sys.exit(main())
