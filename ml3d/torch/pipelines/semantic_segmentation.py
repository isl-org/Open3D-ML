#coding: future_fstrings
import torch, pickle
import torch.nn as nn
import numpy as np
import logging
import sys

from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler

from os.path import exists, join, isfile, dirname, abspath

from .base_pipeline import BasePipeline
from ..dataloaders import TorchDataloader, DefaultBatcher, ConcatBatcher
from ..modules.losses import SemSegLoss
from ..modules.metrics import SemSegMetric
from ...utils import make_dir, LogRecord, Config, PIPELINE
from ...datasets.utils import DataProcessing

logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)

class SemanticSegmentation(BasePipeline):
    def __init__(self,
                model,
                dataset=None,
                name='SemanticSegmentation', 
                batch_size=4,
                val_batch_size=4,
                test_batch_size=3,
                max_epoch=100,  # maximum epoch during training
                learning_rate=1e-2,  # initial learning rate
                lr_decays=0.95,
                save_ckpt_freq=20,
                adam_lr=1e-2,
                scheduler_gamma=0.95,
                main_log_dir='./logs/',
                device='gpu',
                split='train',
                train_sum_dir='train_log',
                **kwargs):
    

        super().__init__(model=model, 
                        dataset=dataset, 
                        name=name,
                        batch_size=batch_size,
                        val_batch_size=val_batch_size,
                        test_batch_size=test_batch_size,
                        max_epoch=max_epoch,
                        learning_rate=learning_rate,
                        lr_decays=lr_decays,
                        save_ckpt_freq=save_ckpt_freq,
                        adam_lr=adam_lr,  
                        scheduler_gamma=scheduler_gamma,
                        main_log_dir=main_log_dir,
                        device=device,
                        split=split,
                        train_sum_dir=train_sum_dir,
                        **kwargs)

        
    def run_inference(self, data):
        cfg = self.cfg
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
                if model.inference_end(inputs, results):
                    break
   
        return model.inference_result

    def run_test(self):
        #self.device = device
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg
        model.device = device
        print(device)
        model.to(device)
        model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = self.get_batcher(device, split='test')

        test_split = TorchDataloader(
            dataset=dataset.get_split('test'),
            preprocess=model.preprocess,
            transform=model.transform,
            use_cache=dataset.cfg.use_cache,
            shuffle=False)

        self.load_ckpt(model.cfg.ckpt_path, False)

        datset_split = self.dataset.get_split('test')

        log.info("Started testing")
        
        with torch.no_grad():
            for idx in tqdm(range(len(test_split)), desc='test'):
                attr = datset_split.get_attr(idx)
                if dataset.is_tested(attr):
                    continue
                data = datset_split.get_data(idx)
                results = self.run_inference(data)
                dataset.save_test_result(results, attr)

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

        Loss = SemSegLoss(self, model, dataset, device)
        Metric = SemSegMetric(self, model, dataset, device)

        batcher = self.get_batcher(device)

        train_split = TorchDataloader(dataset=dataset.get_split('training'),
                                    preprocess=model.preprocess,
                                    transform=model.transform,
                                    use_cache=dataset.cfg.use_cache)

        train_loader = DataLoader(train_split,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  collate_fn=batcher.collate_fn)

        valid_split = TorchDataloader(
            dataset=dataset.get_split('validation'),
            preprocess=model.preprocess,
            transform=model.transform,
            use_cache=dataset.cfg.use_cache)
        valid_loader = DataLoader(
            valid_split,
            batch_size=cfg.val_batch_size,
            shuffle=True,
            collate_fn=batcher.collate_fn)


        self.optimizer, self.scheduler = model.get_optimizer(cfg)

        first_epoch = self.load_ckpt(model.cfg.ckpt_path, True)

        writer = SummaryWriter(join(cfg.logs_dir, cfg.train_sum_dir))

        log.info("Started training")

        for epoch in range(0, cfg.max_epoch + 1):

            print(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            model.train()
            self.losses = []
            self.accs = []
            self.ious = []
            step = 0

            for idx, inputs in enumerate(tqdm(train_loader, desc='training')):

                results = model(inputs['data'])
                loss, gt_labels, predict_scores = model.get_loss(
                    Loss, results, inputs, device)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc = Metric.acc(predict_scores, gt_labels)
                iou = Metric.iou(predict_scores, gt_labels)
                self.losses.append(loss.cpu().item())
                self.accs.append(acc)
                self.ious.append(iou)

                step = step + 1

            self.scheduler.step()

            # --------------------- validation
            model.eval()
            self.valid_losses = []
            self.valid_accs = []
            self.valid_ious = []
            step = 0
            with torch.no_grad():
                for idx, inputs in enumerate(
                        tqdm(valid_loader, desc='validation')):
                    results = model(inputs['data'])
                    loss, gt_labels, predict_scores = model.get_loss(
                        Loss, results, inputs, device)
                    acc = Metric.acc(predict_scores, gt_labels)
                    iou = Metric.iou(predict_scores, gt_labels)

                    self.valid_losses.append(loss.cpu().item())
                    self.valid_accs.append(acc)
                    self.valid_ious.append(iou)

                    step = step + 1

            self.save_logs(writer, epoch)

            if epoch % cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch)

    def get_batcher(self, device, split='training'):

        batcher_name = getattr(self.model.cfg, 'batcher')

        if batcher_name == 'DefaultBatcher':
            batcher = DefaultBatcher()
        elif batcher_name == 'ConcatBatcher':
            batcher = ConcatBatcher(device)
        else:
            batcher = None
        return batcher

    def save_logs(self, writer, epoch):
        accs = np.nanmean(np.array(self.accs), axis=0)
        ious = np.nanmean(np.array(self.ious), axis=0)

        valid_accs = np.nanmean(np.array(self.valid_accs), axis=0)
        valid_ious = np.nanmean(np.array(self.valid_ious), axis=0)

        loss_dict = {
            'Training loss': np.mean(self.losses),
            'Validation loss': np.mean(self.valid_losses)
        }
        acc_dicts = [{
            'Training accuracy': acc,
            'Validation accuracy': val_acc
        } for acc, val_acc in zip(accs, valid_accs)]
        iou_dicts = [{
            'Training IoU': iou,
            'Validation IoU': val_iou
        } for iou, val_iou in zip(ious, valid_ious)]

        # send results to tensorboard
        writer.add_scalars('Loss', loss_dict, epoch)

        for i in range(self.model.cfg.num_classes):
            writer.add_scalars(f'Per-class accuracy/{i+1:02d}', acc_dicts[i],
                               epoch)
            writer.add_scalars(f'Per-class IoU/{i+1:02d}', iou_dicts[i], epoch)

        writer.add_scalars('Overall accuracy', acc_dicts[-1], epoch)
        writer.add_scalars('Mean IoU', iou_dicts[-1], epoch)

        log.info(f"loss train: {loss_dict['Training loss']:.3f} "
                 f" eval: {loss_dict['Validation loss']:.3f}")
        log.info(f"acc train: {acc_dicts[-1]['Training accuracy']:.3f} "
                 f" eval: {acc_dicts[-1]['Validation accuracy']:.3f}")
        log.info(f"iou train: {iou_dicts[-1]['Training IoU']:.3f} "
                 f" eval: {iou_dicts[-1]['Validation IoU']:.3f}")

        # print(acc_dicts[-1])

    def load_ckpt(self, ckpt_path, is_train=True):
        if exists(ckpt_path):
            #path = max(list((cfg.ckpt_path).glob('*.pth')))
            log.info(f'Loading checkpoint {ckpt_path}')
            ckpt = torch.load(ckpt_path)
            first_epoch = ckpt['epoch'] + 1
            self.model.load_state_dict(ckpt['model_state_dict'])
            if is_train:
                if 'optimizer_state_dict' in ckpt:
                    log.info(f'Loading checkpoint optimizer_state_dict')
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    log.info(f'Loading checkpoint scheduler_state_dict')
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            first_epoch = 0
            log.info('No checkpoint')

        return first_epoch

    def save_ckpt(self, epoch):
        path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(path_ckpt)
        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict(),
                 scheduler_state_dict=self.scheduler.state_dict()),
            join(path_ckpt, f'ckpt_{epoch:02d}.pth'))
        log.info(f'Epoch {epoch:3d}: save ckpt to {path_ckpt:s}')

    def filter_valid(self, scores, labels, device):
        valid_scores = scores.reshape(-1, self.model.cfg.num_classes)
        valid_labels = labels.reshape(-1).to(device)

        ignored_bool = torch.zeros_like(valid_labels, dtype=torch.bool)
        for ign_label in self.dataset.cfg.ignored_label_inds:
            ignored_bool = torch.logical_or(ignored_bool,
                                            torch.eq(valid_labels, ign_label))

        valid_idx = torch.where(torch.logical_not(ignored_bool))[0].to(device)

        valid_scores = torch.gather(
            valid_scores, 0,
            valid_idx.unsqueeze(-1).expand(-1, self.model.cfg.num_classes))
        valid_labels = torch.gather(valid_labels, 0, valid_idx)

        # Reduce label values in the range of logit shape
        reducing_list = torch.arange(0,
                                     self.model.cfg.num_classes,
                                     dtype=torch.int64)
        inserted_value = torch.zeros([1], dtype=torch.int64)

        for ign_label in self.dataset.cfg.ignored_label_inds:
            reducing_list = torch.cat([
                reducing_list[:ign_label], inserted_value,
                reducing_list[ign_label:]
            ], 0)
        valid_labels = torch.gather(reducing_list.to(device), 0, valid_labels)

        valid_labels = valid_labels.unsqueeze(0)
        valid_scores = valid_scores.unsqueeze(0).transpose(-2, -1)

        return valid_scores, valid_labels

PIPELINE._register_module(SemanticSegmentation, "torch")
