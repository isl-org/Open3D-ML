import argparse
import copy
import os
import os.path as osp
import time

from ml3d.utils import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('framework', help='deep learning framework: tf or torch')
    parser.add_argument('cfg_file', help='file path to the config file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # parser.add_argument(
    #     '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


from ml3d.datasets import SemanticKITTI

def main():
    args = parse_args()

    cfg = Config.load_from_file(args.cfg_file)

    if args.framework in ['torch', 'pytorch']:
        from ml3d.torch.pipelines import SemanticSegmentation 
        from ml3d.torch.models import RandLANet, KPFCNN
        import torch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    elif args.framework in ['tf', 'tensorflow']:
        from ml3d.tf.pipelines import SemanticSegmentation 
        from ml3d.tf.models import RandLANet, KPFCNN
    else:
        print("Unsupported pipeline {}".format(args.framework.name))
        exit()


    if cfg.model.name in ['RandLANet']:
        model = RandLANet(cfg.model)
    elif cfg.model.name in ['KPFCNN', 'KPConv']:
        model = KPFCNN(cfg.model)
    else:
        print("Unsupported model {}".format(args.model.name))
        exit()

    if cfg.dataset.name in ['SemanticKITTI']:
        dataset = SemanticKITTI(cfg.dataset)
    else:
        print("Unsupported datasets {}".format(args.dataset.name))

    # TODO choose device
    
    pipeline = SemanticSegmentation(model, dataset, cfg.pipeline)
    pipeline.run_train(device)

    # log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # # init the meta dict to record some important information such as
    # # environment info and seed, which will be logged
    # meta = dict()
    # # log env info
    # env_info_dict = collect_env()
    # env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    # dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' +
    #             dash_line)
    # meta['env_info'] = env_info

    # # log some basic info
    # logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # # set random seeds
    # if args.seed is not None:
    #     logger.info(f'Set random seed to {args.seed}, '
    #                 f'deterministic: {args.deterministic}')
    #     set_random_seed(args.seed, deterministic=args.deterministic)
    # cfg.seed = args.seed
    # meta['seed'] = args.seed

    # model = build_detector(
    #     cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # datasets = [build_dataset(cfg.data.train)]
    # if len(cfg.workflow) == 2:
    #     val_dataset = copy.deepcopy(cfg.data.val)
    #     val_dataset.pipeline = cfg.data.train.pipeline
    #     datasets.append(build_dataset(val_dataset))
    # if cfg.checkpoint_config is not None:
    #     # save mmdet version, config file content and class names in
    #     # checkpoints as meta data
    #     cfg.checkpoint_config.meta = dict(
    #         mmdet_version=__version__,
    #         config=cfg.pretty_text,
    #         CLASSES=datasets[0].CLASSES)
    # # add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES
    # train_detector(
    #     model,
    #     datasets,
    #     cfg,
    #     distributed=distributed,
    #     validate=(not args.no_validate),
    #     timestamp=timestamp,
    #     meta=meta)


if __name__ == '__main__':
    main()
