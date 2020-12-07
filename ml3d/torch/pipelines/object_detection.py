import torch
import logging
from tqdm import tqdm

from datetime import datetime

from os.path import exists, join
from torch.utils.data import DataLoader
from pathlib import Path

from .base_pipeline import BasePipeline
from ..dataloaders import TorchDataloader
from torch.utils.tensorboard import SummaryWriter
from ..utils import latest_torch_ckpt
from ...utils import make_dir, PIPELINE, LogRecord, get_runid

logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class ObjectDetection(BasePipeline):
    """
    Pipeline for object detection. 
    """

    def __init__(self,
                 model,
                 dataset=None,
                 name='ObjectDetection',
                 main_log_dir='./logs/',
                 device='gpu',
                 split='train',
                 **kwargs):
        super().__init__(model=model,
                         dataset=dataset,
                         name=name,
                         main_log_dir=main_log_dir,
                         device=device,
                         split=split,
                         **kwargs)

    def run_inference(self, data):
        """
        Run inference on a given data.

        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """
        model = self.model
        device = self.device

        model.to(device)
        model.device = device
        model.eval()

        model.inference_begin(data)

        with torch.no_grad():
            while True:
                inputs = model.inference_preprocess()
                results = model(inputs['data'])

                print(model.get_loss(None, results, inputs['data']))
                if model.inference_end(inputs, results):
                    break

        return model.inference_result

    def run_test(self):
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg
        model.device = device
        model.to(device)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        test_split = TorchDataloader(dataset=dataset.get_split('test'),
                                     preprocess=model.preprocess,
                                     transform=model.transform,
                                     use_cache=dataset.cfg.use_cache,
                                     shuffle=False)

        self.load_ckpt(model.cfg.ckpt_path)

        log.info("Started testing")

        results = []
        with torch.no_grad():
            for idx in tqdm(range(len(test_split)), desc='test'):
                data = test_split[idx]
                result = self.run_inference(data['data'])
                results.extend(result)

    def run_train(self):
        model = self.model
        device = self.device
        model.device = device
        dataset = self.dataset

        cfg = self.cfg
        model.to(device)

        log.info("DEVICE : {}".format(device))
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        train_dataset = dataset.get_split('training')
        train_split = TorchDataloader(dataset=train_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      use_cache=dataset.cfg.use_cache,
                                      steps_per_epoch=dataset.cfg.get(
                                          'steps_per_epoch_train', None))
        train_loader = DataLoader(train_split,
                                  batch_size=cfg.batch_size)

        valid_dataset = dataset.get_split('validation')
        valid_split = TorchDataloader(dataset=valid_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      use_cache=dataset.cfg.use_cache,
                                      steps_per_epoch=dataset.cfg.get(
                                          'steps_per_epoch_valid', None))
        valid_loader = DataLoader(valid_split,
                                  batch_size=cfg.val_batch_size)

        self.optimizer, self.scheduler = model.get_optimizer(cfg.optimizer)

        is_resume = model.cfg.get('is_resume', True)
        self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

        dataset_name = dataset.name if dataset is not None else ''
        tensorboard_dir = join(
            self.cfg.train_sum_dir,
            model.__class__.__name__ + '_' + dataset_name + '_torch')
        runid = get_runid(tensorboard_dir)
        self.tensorboard_dir = join(self.cfg.train_sum_dir,
                                    runid + '_' + Path(tensorboard_dir).name)

        writer = SummaryWriter(self.tensorboard_dir)
        self.save_config(writer)
        log.info("Writing summary in {}.".format(self.tensorboard_dir))

        log.info("Started training")

        for epoch in range(0, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            model.train()
            self.losses = []
            #self.accs = []
            #self.ious = []

            for step, inputs in enumerate(tqdm(train_loader, desc='training')):
                results = model(inputs['data'])
                loss = model.loss(results, inputs)

                self.optimizer.zero_grad()
                loss.backward()
                if model.cfg.get('grad_clip_norm', -1) > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                                                    model.cfg.grad_clip_norm)
                self.optimizer.step()

                #acc = metric.acc(predict_scores, gt_labels)
                #iou = metric.iou(predict_scores, gt_labels)

                self.losses.append(loss.cpu().item())
                #self.accs.append(acc)
                #self.ious.append(iou)

            #self.scheduler.step()

            # --------------------- validation
            #model.eval()
            #self.run_valid()

            self.save_logs(writer, epoch)

            if epoch % cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch)


    def load_ckpt(self, ckpt_path=None, is_resume=True):
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)

        if ckpt_path is None:
            ckpt_path = latest_torch_ckpt(train_ckpt_dir)
            if ckpt_path is not None and is_resume:
                log.info('ckpt_path not given. Restore from the latest ckpt')
            else:
                log.info('Initializing from scratch.')
                return

        if not exists(ckpt_path):
            raise FileNotFoundError(f' ckpt {ckpt_path} not found')

        log.info(f'Loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            log.info(f'Loading checkpoint optimizer_state_dict')
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler'):
            log.info(f'Loading checkpoint scheduler_state_dict')
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    def save_ckpt(self, epoch):
        path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(path_ckpt)
        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict()),
                 #optimizer_state_dict=self.optimizer.state_dict(),
                 #scheduler_state_dict=self.scheduler.state_dict()),
            join(path_ckpt, f'ckpt_{epoch:05d}.pth'))
        log.info(f'Epoch {epoch:3d}: save ckpt to {path_ckpt:s}')


PIPELINE._register_module(ObjectDetection, "torch")
