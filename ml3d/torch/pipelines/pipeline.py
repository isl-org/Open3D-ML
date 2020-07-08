import warnings
import torch

from ml3d.module import (Runner, build_network, build_optimizer)
# from ml3d.module import (Runner, build_network, build_dataset, build_dataloader)
from ml3d.module.misc import Composer
from ml3d.util import load_checkpoint, parse_losses


def batch_processor(model, data, train_mode):
    """Process a data batch.

    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.

    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.

    Returns:
        dict: A dict containing losses and log vars.
    """
    
    if (train_mode):
        result  = model(*data[0])
        losses  = model.compute_loss(result, *data[1])
        
        loss, log_vars = parse_losses(losses)

        outputs = dict(loss=loss)
    # outputs = dict(
    #     loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs

class Predictor():
    """
    Predictor.
    """
    def __init__(self, cfg, checkpoint=None, device='cuda:0'):
        self.cfg     = cfg
        self.network = build_network(self.cfg.cfg_dict['network'])
        if checkpoint is not None:
            if device is 'cpu':
                checkpoint = load_checkpoint(self.network, checkpoint, torch.device(device))
            else:
                checkpoint = load_checkpoint(self.network, checkpoint, device)
        else :
            warnings.warn('No checkpoint file')

        self.network.to(device)
        pipeline = [self.network]
        self._pipeline = Composer(pipeline)

    def __call__(self, input_data):
        # print(input_data)
        # print(self._pipeline)
        with torch.no_grad():
            #result = self.network(input_data)
            result = self._pipeline(input_data)

        return result

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, cfg, data_loaders):

        self.cfg = cfg


        self.network  = build_network(cfg.network)
        
        # self.datasets = build_dataset(cfg.cfg_dict['data']['train'])
        
        # if cfg.cfg_dict['checkpoint_config'] is not None:
        #     # save mmdet version, config file content and class names in
        #     # checkpoints as meta data
        #     cfg.checkpoint_config.meta = dict(
        #         mmdet_version=__version__,
        #         config=cfg.pretty_text,
        #         CLASSES=datasets[0].CLASSES)


        self.data_loaders = data_loaders
        # self.data_loaders = [
        # build_dataloader(
        #     ds,
        #     cfg.data.samples_per_gpu,
        #     cfg.data.workers_per_gpu,
        #     # cfg.gpus will be ignored if distributed
        #     len(cfg.gpu_ids),
        #     dist=distributed,
        #     seed=cfg.seed) for ds in self.dataset
        # ]

        optimizer = build_optimizer(self.network, cfg.optimizer)
        #optimizer = None

        self.runner = Runner(
        self.network,
        batch_processor,
        optimizer,
        cfg.work_dir,)

        self.runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    def train(self):
        self.runner.train(self.data_loaders)

        
