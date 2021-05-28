import torch, pickle
import torch.nn as nn
import numpy as np
import logging
import sys

from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, IterableDataset, DataLoader
from pathlib import Path
from sklearn.metrics import confusion_matrix

from os.path import exists, join, isfile, dirname, abspath

from .base_pipeline import BasePipeline
from ..dataloaders import get_sampler, TorchDataloader, DefaultBatcher, ConcatBatcher
from ..utils import latest_torch_ckpt
from ..modules.losses import SemSegLoss
from ..modules.metrics import SemSegMetric
from ...utils import make_dir, LogRecord, Config, PIPELINE, get_runid, code2md
from ...datasets.utils import DataProcessing
from ...datasets import InferenceDummySplit

logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class SemanticSegmentation(BasePipeline):
    """This class allows you to perform semantic segmentation for both training
    and inference using the Torch. This pipeline has multiple stages: Pre-
    processing, loading dataset, testing, and inference or training.

    **Example:**
        This example loads the Semantic Segmentation and performs a training using the SemanticKITTI dataset.

            import torch, pickle
            import torch.nn as nn

            from .base_pipeline import BasePipeline
            from torch.utils.tensorboard import SummaryWriter
            from ..dataloaders import get_sampler, TorchDataloader, DefaultBatcher, ConcatBatcher

            Mydataset = TorchDataloader(dataset=dataset.get_split('training')),
            MyModel = SemanticSegmentation(self,model,dataset=Mydataset, name='SemanticSegmentation',
            name='MySemanticSegmentation',
            batch_size=4,
            val_batch_size=4,
            test_batch_size=3,
            max_epoch=100,
            learning_rate=1e-2,
            lr_decays=0.95,
            save_ckpt_freq=20,
            adam_lr=1e-2,
            scheduler_gamma=0.95,
            momentum=0.98,
            main_log_dir='./logs/',
            device='gpu',
            split='train',
            train_sum_dir='train_log')

    **Args:**
            dataset: The 3D ML dataset class. You can use the base dataset, sample datasets , or a custom dataset.
            model: The model to be used for building the pipeline.
            name: The name of the current training.
            batch_size: The batch size to be used for training.
            val_batch_size: The batch size to be used for validation.
            test_batch_size: The batch size to be used for testing.
            max_epoch: The maximum size of the epoch to be used for training.
            leanring_rate: The hyperparameter that controls the weights during training. Also, known as step size.
            lr_decays: The learning rate decay for the training.
            save_ckpt_freq: The frequency in which the checkpoint should be saved.
            adam_lr: The leanring rate to be applied for Adam optimization.
            scheduler_gamma: The decaying factor associated with the scheduler.
            momentum: The momentum that accelerates the training rate schedule.
            main_log_dir: The directory where logs are stored.
            device: The device to be used for training.
            split: The dataset split to be used. In this example, we have used "train".
            train_sum_dir: The directory where the trainig summary is stored.

    **Returns:**
            class: The corresponding class.
    """

    def __init__(
            self,
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
            momentum=0.98,
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
                         momentum=momentum,
                         main_log_dir=main_log_dir,
                         device=device,
                         split=split,
                         train_sum_dir=train_sum_dir,
                         **kwargs)
        """
        Run inference on given data.

        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """

    def run_inference(self, data):
        cfg = self.cfg
        model = self.model
        device = self.device

        model.to(device)
        model.device = device
        model.eval()

        batcher = self.get_batcher(device)
        infer_dataset = InferenceDummySplit(data)
        self.dataset_split = infer_dataset
        infer_sampler = infer_dataset.sampler
        infer_split = TorchDataloader(dataset=infer_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      sampler=infer_sampler,
                                      use_cache=False)
        infer_loader = DataLoader(infer_split,
                                  batch_size=cfg.batch_size,
                                  sampler=get_sampler(infer_sampler),
                                  collate_fn=batcher.collate_fn)

        model.trans_point_sampler = infer_sampler.get_point_sampler()
        self.curr_cloud_id = -1
        self.test_probs = []
        self.test_labels = []
        self.ori_test_probs = []
        self.ori_test_labels = []

        with torch.no_grad():
            for step, inputs in enumerate(infer_loader):
                results = model(inputs['data'])
                self.update_tests(infer_sampler, inputs, results)

        inference_result = {
            'predict_labels': self.ori_test_labels.pop(),
            'predict_scores': self.ori_test_probs.pop()
        }

        return inference_result

    """
    Run the test using the data passed.
    
    """

    def run_test(self):
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg
        model.device = device
        model.to(device)
        model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        metric = SemSegMetric()

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = self.get_batcher(device)

        test_dataset = dataset.get_split('test')
        test_sampler = test_dataset.sampler
        test_split = TorchDataloader(dataset=test_dataset,
                                     preprocess=model.preprocess,
                                     transform=model.transform,
                                     sampler=test_sampler,
                                     use_cache=dataset.cfg.use_cache)
        test_loader = DataLoader(test_split,
                                 batch_size=cfg.test_batch_size,
                                 sampler=get_sampler(test_sampler),
                                 collate_fn=batcher.collate_fn)

        self.dataset_split = test_dataset

        self.load_ckpt(model.cfg.ckpt_path)

        model.trans_point_sampler = test_sampler.get_point_sampler()
        self.curr_cloud_id = -1
        self.test_probs = []
        self.test_labels = []
        self.ori_test_probs = []
        self.ori_test_labels = []

        log.info("Started testing")

        with torch.no_grad():
            for step, inputs in enumerate(test_loader):
                if hasattr(inputs['data'], 'to'):
                    inputs['data'].to(device)
                results = model(inputs['data'])
                self.update_tests(test_sampler, inputs, results)

                if self.complete_infer:
                    inference_result = {
                        'predict_labels': self.ori_test_labels.pop(),
                        'predict_scores': self.ori_test_probs.pop()
                    }
                    attr = self.dataset_split.get_attr(test_sampler.cloud_id)
                    dataset.save_test_result(inference_result, attr)

        log.info("Finshed testing")

    """
    Update tests using sampler, inputs, and results.
    
    """

    def update_tests(self, sampler, inputs, results):
        split = sampler.split
        end_threshold = 0.5
        if self.curr_cloud_id != sampler.cloud_id:
            self.curr_cloud_id = sampler.cloud_id
            num_points = sampler.possibilities[sampler.cloud_id].shape[0]
            self.pbar = tqdm(total=num_points,
                             desc="{} {}/{}".format(split, self.curr_cloud_id,
                                                    len(sampler.dataset)))
            self.pbar_update = 0
            self.test_probs.append(
                np.zeros(shape=[num_points, self.model.cfg.num_classes],
                         dtype=np.float16))
            self.test_labels.append(np.zeros(shape=[num_points],
                                             dtype=np.int16))
            self.complete_infer = False

        this_possiblility = sampler.possibilities[sampler.cloud_id]
        self.pbar.update(this_possiblility[this_possiblility > end_threshold].shape[0] \
            - self.pbar_update)
        self.pbar_update = this_possiblility[
            this_possiblility > end_threshold].shape[0]
        self.test_probs[self.curr_cloud_id], self.test_labels[self.curr_cloud_id] \
            = self.model.update_probs(inputs, results,
                self.test_probs[self.curr_cloud_id],
                self.test_labels[self.curr_cloud_id])

        if split in ['test'] and this_possiblility[this_possiblility > end_threshold].shape[0] \
          == this_possiblility.shape[0]:

            proj_inds = self.model.preprocess(
                self.dataset_split.get_data(self.curr_cloud_id), {
                    'split': split
                }).get('proj_inds', None)
            if proj_inds is None:
                proj_inds = np.arange(
                    self.test_probs[self.curr_cloud_id].shape[0])
            self.ori_test_probs.append(
                self.test_probs[self.curr_cloud_id][proj_inds])
            self.ori_test_labels.append(
                self.test_labels[self.curr_cloud_id][proj_inds])
            self.complete_infer = True

    """
    Run the training on the self model.
    
    """

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
        self.metric_train = SemSegMetric()
        self.metric_val = SemSegMetric()

        self.batcher = self.get_batcher(device)

        train_dataset = dataset.get_split('train')
        train_sampler = train_dataset.sampler
        train_split = TorchDataloader(dataset=train_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      sampler=train_sampler,
                                      use_cache=dataset.cfg.use_cache,
                                      steps_per_epoch=dataset.cfg.get(
                                          'steps_per_epoch_train', None))

        train_loader = DataLoader(
            train_split,
            batch_size=cfg.batch_size,
            sampler=get_sampler(train_sampler),
            num_workers=cfg.get('num_workers', 2),
            pin_memory=cfg.get('pin_memory', True),
            collate_fn=self.batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed))
        )  # numpy expects np.uint32, whereas torch returns np.uint64.

        valid_dataset = dataset.get_split('validation')
        valid_sampler = valid_dataset.sampler
        valid_split = TorchDataloader(dataset=valid_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      sampler=valid_sampler,
                                      use_cache=dataset.cfg.use_cache,
                                      steps_per_epoch=dataset.cfg.get(
                                          'steps_per_epoch_valid', None))

        valid_loader = DataLoader(
            valid_split,
            batch_size=cfg.val_batch_size,
            sampler=get_sampler(valid_sampler),
            num_workers=cfg.get('num_workers', 2),
            pin_memory=cfg.get('pin_memory', True),
            collate_fn=self.batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed)))

        self.optimizer, self.scheduler = model.get_optimizer(cfg)

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
            self.metric_train.reset()
            self.metric_val.reset()
            self.losses = []
            model.trans_point_sampler = train_sampler.get_point_sampler()

            for step, inputs in enumerate(tqdm(train_loader, desc='training')):
                if hasattr(inputs['data'], 'to'):
                    inputs['data'].to(device)
                self.optimizer.zero_grad()
                results = model(inputs['data'])
                loss, gt_labels, predict_scores = model.get_loss(
                    Loss, results, inputs, device)

                if predict_scores.size()[-1] == 0:
                    continue

                loss.backward()
                if model.cfg.get('grad_clip_norm', -1) > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                                                    model.cfg.grad_clip_norm)
                self.optimizer.step()

                self.metric_train.update(predict_scores, gt_labels)

                self.losses.append(loss.cpu().item())

            self.scheduler.step()

            # --------------------- validation
            model.eval()
            self.valid_losses = []
            model.trans_point_sampler = valid_sampler.get_point_sampler()

            with torch.no_grad():
                for step, inputs in enumerate(
                        tqdm(valid_loader, desc='validation')):
                    if hasattr(inputs['data'], 'to'):
                        inputs['data'].to(device)

                    results = model(inputs['data'])
                    loss, gt_labels, predict_scores = model.get_loss(
                        Loss, results, inputs, device)

                    if predict_scores.size()[-1] == 0:
                        continue

                    self.metric_val.update(predict_scores, gt_labels)

                    self.valid_losses.append(loss.cpu().item())

            self.save_logs(writer, epoch)

            if epoch % cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch)

    """
    Get the batcher to be used based on the device and split.
    
    """

    def get_batcher(self, device, split='training'):

        batcher_name = getattr(self.model.cfg, 'batcher')

        if batcher_name == 'DefaultBatcher':
            batcher = DefaultBatcher()
        elif batcher_name == 'ConcatBatcher':
            batcher = ConcatBatcher(device, self.model.cfg.name)
        else:
            batcher = None
        return batcher

    """
    Save logs from the training and send results to TensorBoard.
    
    """

    def save_logs(self, writer, epoch):

        train_accs = self.metric_train.acc()
        val_accs = self.metric_val.acc()

        train_ious = self.metric_train.iou()
        val_ious = self.metric_val.iou()

        loss_dict = {
            'Training loss': np.mean(self.losses),
            'Validation loss': np.mean(self.valid_losses)
        }
        acc_dicts = [{
            'Training accuracy': acc,
            'Validation accuracy': val_acc
        } for acc, val_acc in zip(train_accs, val_accs)]

        iou_dicts = [{
            'Training IoU': iou,
            'Validation IoU': val_iou
        } for iou, val_iou in zip(train_ious, val_ious)]

        for key, val in loss_dict.items():
            writer.add_scalar(key, val, epoch)
        for key, val in acc_dicts[-1].items():
            writer.add_scalar("{}/ Overall".format(key), val, epoch)
        for key, val in iou_dicts[-1].items():
            writer.add_scalar("{}/ Overall".format(key), val, epoch)

        log.info(f"Loss train: {loss_dict['Training loss']:.3f} "
                 f" eval: {loss_dict['Validation loss']:.3f}")
        log.info(f"Mean acc train: {acc_dicts[-1]['Training accuracy']:.3f} "
                 f" eval: {acc_dicts[-1]['Validation accuracy']:.3f}")
        log.info(f"Mean IoU train: {iou_dicts[-1]['Training IoU']:.3f} "
                 f" eval: {iou_dicts[-1]['Validation IoU']:.3f}")

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        """Load a checkpoint. You must pass the checkpoint and indicate if you want to resume."""
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

    """
    Save a checkpoint at the passed epoch.
    
    """

    def save_ckpt(self, epoch):
        path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(path_ckpt)
        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict(),
                 scheduler_state_dict=self.scheduler.state_dict()),
            join(path_ckpt, f'ckpt_{epoch:05d}.pth'))
        log.info(f'Epoch {epoch:3d}: save ckpt to {path_ckpt:s}')

    """
    Save experiment configuration with Torch summary.
    
    """

    def save_config(self, writer):
        writer.add_text("Description/Open3D-ML", self.cfg_tb['readme'], 0)
        writer.add_text("Description/Command line", self.cfg_tb['cmd_line'], 0)
        writer.add_text('Configuration/Dataset',
                        code2md(self.cfg_tb['dataset'], language='json'), 0)
        writer.add_text('Configuration/Model',
                        code2md(self.cfg_tb['model'], language='json'), 0)
        writer.add_text('Configuration/Pipeline',
                        code2md(self.cfg_tb['pipeline'], language='json'), 0)


PIPELINE._register_module(SemanticSegmentation, "torch")
