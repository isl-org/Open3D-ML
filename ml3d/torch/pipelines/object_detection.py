import logging
import re
import numpy as np
import torch
import torch.distributed as dist

from datetime import datetime
from os.path import exists, join
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from .base_pipeline import BasePipeline
from ..dataloaders import TorchDataloader, ConcatBatcher
from torch.utils.tensorboard import SummaryWriter
# pylint: disable-next=unused-import
from open3d.visualization.tensorboard_plugin import summary
from ..utils import latest_torch_ckpt
from ...utils import make_dir, PIPELINE, get_runid, code2md
from ...datasets.utils import BEVBox3D

from ...metrics.mAP import mAP

log = logging.getLogger(__name__)


class ObjectDetection(BasePipeline):
    """Pipeline for object detection."""

    def __init__(self,
                 model,
                 dataset=None,
                 name='ObjectDetection',
                 main_log_dir='./logs/',
                 device='cuda',
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
        """Run inference on given data.

        Args:
            data: A raw data.

        Returns:
            Returns the inference results.
        """
        model = self.model

        model.eval()

        # If run_inference is called on raw data.
        if isinstance(data, dict):
            batcher = ConcatBatcher(self.device, model.cfg.name)
            data = batcher.collate_fn([{
                'data': data,
                'attr': {
                    'split': 'test'
                }
            }])

        data.to(self.device)

        with torch.no_grad():
            results = model(data)
            boxes = model.inference_end(results, data)

        return boxes

    def run_test(self):
        """Run test with test data split, computes mean average precision of the
        prediction results.
        """
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg

        model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(device, model.cfg.name)

        test_split = TorchDataloader(dataset=dataset.get_split('test'),
                                     preprocess=model.preprocess,
                                     transform=model.transform,
                                     use_cache=False,
                                     shuffle=False)
        test_loader = DataLoader(
            test_split,
            batch_size=cfg.test_batch_size,
            num_workers=cfg.get('num_workers', 4),
            pin_memory=cfg.get('pin_memory', True),
            collate_fn=batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed)))

        self.load_ckpt(model.cfg.ckpt_path)

        if cfg.get('test_compute_metric', True):
            self.run_valid()

        log.info("Started testing")
        self.test_ious = []
        pred = []
        record_summary = 'test' in cfg.get('summary').get('record_for', [])
        with torch.no_grad():
            for data in tqdm(test_loader, desc='testing'):
                results = self.run_inference(data)
                pred.extend(results)
                dataset.save_test_result(results, data.attr)
                # Save only for the first batch
                if record_summary and 'test' not in self.summary:
                    boxes = results  # inference_end already executed in run_inference
                    self.summary['test'] = self.get_3d_summary(boxes,
                                                               data,
                                                               0,
                                                               save_gt=False)

    def run_valid(self, epoch=0):
        """Run validation with validation data split, computes mean average
        precision and the loss of the prediction results.

        Args:
            epoch (int): step for TensorBoard summary. Defaults to 0 if
                unspecified.
        """
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg

        model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_valid_' + timestamp + '.txt')
        if self.rank == 0:
            log.info("Logging in file : {}".format(log_file_path))
            log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(device, model.cfg.name)

        valid_dataset = dataset.get_split('validation')
        valid_split = TorchDataloader(dataset=valid_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      use_cache=dataset.cfg.use_cache,
                                      shuffle=True,
                                      steps_per_epoch=dataset.cfg.get(
                                          'steps_per_epoch_valid', None))

        if self.distributed:
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_split)
        else:
            valid_sampler = None

        valid_loader = DataLoader(valid_split,
                                  batch_size=cfg.val_batch_size,
                                  num_workers=cfg.get('num_workers', 0),
                                  pin_memory=cfg.get('pin_memory', False),
                                  collate_fn=batcher.collate_fn,
                                  sampler=valid_sampler)

        record_summary = self.rank == 0 and 'valid' in cfg.get('summary').get(
            'record_for', [])
        log.info("Started validation")

        self.valid_losses = {}

        pred = []
        gt = []
        with torch.no_grad():
            for data in tqdm(valid_loader, desc='validation'):
                data.to(device)
                results = model(data)
                loss = model.get_loss(results, data)
                for l, v in loss.items():
                    if l not in self.valid_losses:
                        self.valid_losses[l] = []
                    self.valid_losses[l].append(v.cpu().numpy())

                # convert to bboxes for mAP evaluation
                boxes = model.inference_end(results, data)
                pred.extend([BEVBox3D.to_dicts(b) for b in boxes])
                gt.extend([BEVBox3D.to_dicts(b) for b in data.bbox_objs])
                if record_summary:
                    self.summary['valid'] = self.get_3d_summary(boxes,
                                                                data,
                                                                epoch,
                                                                results=results)
                record_summary = False  # Save only for the first batch

        sum_loss = 0
        desc = "validation - "
        for l, v in self.valid_losses.items():
            desc += " %s: %.03f" % (l, np.mean(v))
            sum_loss += np.mean(v)
        desc += " > loss: %.03f" % sum_loss

        log.info(desc)

        overlaps = cfg.get("overlaps", [0.5])
        similar_classes = cfg.get("similar_classes", {})
        difficulties = cfg.get("difficulties", [0])

        if self.distributed:
            gt_gather = [None for _ in range(dist.get_world_size())]
            pred_gather = [None for _ in range(dist.get_world_size())]

            dist.gather_object(gt, gt_gather if self.rank == 0 else None, dst=0)
            dist.gather_object(pred,
                               pred_gather if self.rank == 0 else None,
                               dst=0)

            if self.rank == 0:
                gt = sum(gt_gather, [])
                pred = sum(pred_gather, [])

        if self.rank != 0:
            return

        ap = mAP(pred,
                 gt,
                 model.classes,
                 difficulties,
                 overlaps,
                 similar_classes=similar_classes)

        log.info("")
        log.info("=============== mAP BEV ===============")
        log.info(("class \\ difficulty  " +
                  "{:>5} " * len(difficulties)).format(*difficulties))
        for i, c in enumerate(model.classes):
            log.info(("{:<20} " + "{:>5.2f} " * len(difficulties)).format(
                c + ":", *ap[i, :, 0]))
        log.info("Overall: {:.2f}".format(np.mean(ap[:, -1])))
        self.valid_losses["mAP BEV"] = np.mean(ap[:, -1])

        ap = mAP(pred,
                 gt,
                 model.classes,
                 difficulties,
                 overlaps,
                 similar_classes=similar_classes,
                 bev=False)
        log.info("")
        log.info("=============== mAP  3D ===============")
        log.info(("class \\ difficulty  " +
                  "{:>5} " * len(difficulties)).format(*difficulties))
        for i, c in enumerate(model.classes):
            log.info(("{:<20} " + "{:>5.2f} " * len(difficulties)).format(
                c + ":", *ap[i, :, 0]))
        log.info("Overall: {:.2f}".format(np.mean(ap[:, -1])))
        self.valid_losses["mAP 3D"] = np.mean(ap[:, -1])

    def run_train(self):
        """Run training with train data split."""
        torch.manual_seed(self.rng.integers(np.iinfo(
            np.int32).max))  # Random reproducible seed for torch
        rank = self.rank  # Rank for distributed training
        model = self.model
        device = self.device
        dataset = self.dataset

        cfg = self.cfg

        if rank == 0:
            log.info("DEVICE : {}".format(device))
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            log_file_path = join(cfg.logs_dir,
                                 'log_train_' + timestamp + '.txt')
            log.info("Logging in file : {}".format(log_file_path))
            log.addHandler(logging.FileHandler(log_file_path))

        batcher = ConcatBatcher(device, model.cfg.name)

        train_dataset = dataset.get_split('training')
        train_split = TorchDataloader(dataset=train_dataset,
                                      preprocess=model.preprocess,
                                      transform=model.transform,
                                      use_cache=dataset.cfg.use_cache,
                                      steps_per_epoch=dataset.cfg.get(
                                          'steps_per_epoch_train', None))

        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_split)
        else:
            train_sampler = None

        train_loader = DataLoader(
            train_split,
            batch_size=cfg.batch_size,
            num_workers=cfg.get('num_workers', 0),
            pin_memory=cfg.get('pin_memory', False),
            collate_fn=batcher.collate_fn,
            sampler=train_sampler,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed))
        )  # numpy expects np.uint32, whereas torch returns np.uint64.

        self.optimizer, self.scheduler = model.get_optimizer(cfg.optimizer)

        is_resume = model.cfg.get('is_resume', True)
        start_ep = self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

        dataset_name = dataset.name if dataset is not None else ''
        tensorboard_dir = join(
            self.cfg.train_sum_dir,
            model.__class__.__name__ + '_' + dataset_name + '_torch')
        runid = get_runid(tensorboard_dir)
        self.tensorboard_dir = join(cfg.train_sum_dir,
                                    runid + '_' + Path(tensorboard_dir).name)

        writer = SummaryWriter(self.tensorboard_dir)
        if rank == 0:
            self.save_config(writer)
            log.info("Writing summary in {}.".format(self.tensorboard_dir))

        # wrap model for multiple GPU
        if self.distributed:
            model.cuda(self.device)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.device])
            model.get_loss = model.module.get_loss
            model.cfg = model.module.cfg
            model.inference_end = model.module.inference_end

        record_summary = self.rank == 0 and 'train' in cfg.get('summary').get(
            'record_for', [])

        if rank == 0:
            log.info("Started training")

        for epoch in range(start_ep, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            if self.distributed:
                train_sampler.set_epoch(epoch)

            model.train()
            self.losses = {}

            process_bar = tqdm(train_loader, desc='training')
            for data in process_bar:
                data.to(device)
                results = model(data)
                loss = model.get_loss(results, data)
                loss_sum = sum(loss.values())

                self.optimizer.zero_grad()
                loss_sum.backward()
                if self.distributed:
                    if model.module.cfg.get('grad_clip_norm', -1) > 0:
                        torch.nn.utils.clip_grad_value_(
                            model.module.parameters(),
                            model.module.cfg.grad_clip_norm)
                else:
                    if model.cfg.get('grad_clip_norm', -1) > 0:
                        torch.nn.utils.clip_grad_value_(
                            model.parameters(), model.cfg.grad_clip_norm)

                self.optimizer.step()

                # Record visualization for the last iteration
                if record_summary and process_bar.n == process_bar.total - 1:
                    boxes = model.inference_end(results, data)
                    self.summary['train'] = self.get_3d_summary(boxes,
                                                                data,
                                                                epoch,
                                                                results=results)
                desc = "training - "
                for l, v in loss.items():
                    if l not in self.losses:
                        self.losses[l] = []
                    self.losses[l].append(v.cpu().detach().numpy())
                    desc += " %s: %.03f" % (l, v.cpu().detach().numpy())
                desc += " > loss: %.03f" % loss_sum.cpu().detach().numpy()
                process_bar.set_description(desc)
                process_bar.refresh()

                if self.distributed:
                    dist.barrier()

            if self.scheduler is not None:
                self.scheduler.step()

            # --------------------- validation
            if epoch % cfg.get("validation_freq", 1) == 0:
                self.run_valid()
                if self.distributed:
                    dist.barrier()

            if rank == 0:
                self.save_logs(writer, epoch)
                if epoch % cfg.save_ckpt_freq == 0 or epoch == cfg.max_epoch:
                    self.save_ckpt(epoch)

    def get_3d_summary(self,
                       infer_bboxes_batch,
                       inputs_batch,
                       epoch,
                       results=None,
                       save_gt=True):
        """
        Create visualization for input point cloud and network output bounding
        boxes.

        Args:
            infer_bboxes_batch (Sequence[Sequence[BoundingBox3D]): Batch of
                predicted bounding boxes from inference_end()
            inputs_batch (Sequence[Sequence[bbox_objs: Object3D, point:
                array(N,3)]]): Batch of ground truth boxes and pointclouds.
            epoch (int): step
            results (torch.FloatTensor): Model output (only required for RPN
                stage of PointRCNN).
            save_gt (bool): Save ground truth (for 'train' or 'valid' stages).

        Returns:
            [Dict] visualizations of inputs and outputs suitable to save as an
                Open3D for TensorBoard summary.
        """
        if not hasattr(self, "_first_step"):
            self._first_step = epoch
        if not hasattr(self.dataset, "name_to_labels"):
            self.dataset.name_to_labels = {
                name: label
                for label, name in self.dataset.get_label_to_names().items()
            }
        cfg = self.cfg.get('summary')
        max_pts = cfg.get('max_pts')
        if max_pts is None:
            max_pts = np.iinfo(np.int32).max
        use_reference = cfg.get('use_reference', False)
        max_outputs = min(cfg.get('max_outputs', 1), len(inputs_batch.point))
        input_pcd = []

        if self.model.cfg['name'] == 'PointRCNN' and self.model.mode == 'RPN':
            for pointcloud in inputs_batch.point[:max_outputs]:
                if self._first_step == epoch or not use_reference:
                    pcd_step = int(
                        np.ceil(pointcloud.shape[0] /
                                min(max_pts, pointcloud.shape[0])))
                    pcd = pointcloud[::pcd_step, :3].cpu().detach()
                    input_pcd.append(pcd)
            rpn_scores_norm = torch.sigmoid(results['cls'])
            seg_mask = (rpn_scores_norm > self.model.score_thres).float()
            cls_score = [ten.cpu().detach() for ten in seg_mask]
            summary3d = {
                'input_pointcloud': {
                    "vertex_positions":
                        input_pcd if self._first_step == epoch or
                        not use_reference else self._first_step,
                    "vertex_predict_labels":
                        cls_score,
                    "label_to_names": {
                        0: "background",
                        1: "foreground"
                    }
                }
            }
            if save_gt:
                log.warning("Saving ground truth not supported for PointRCNN "
                            "in RPN mode")
            return summary3d

        inputs_batch_gt_bboxes = (inputs_batch.bbox_objs[:max_outputs]
                                  if save_gt else ([],) * max_outputs)
        for infer_bboxes, gt_bboxes, pointcloud in zip(
                infer_bboxes_batch[:max_outputs], inputs_batch_gt_bboxes,
                inputs_batch.point[:max_outputs]):
            for bboxes in (gt_bboxes, infer_bboxes):
                for bb in bboxes:  # LUT needs label_class to be int, not str
                    if not isinstance(bb.label_class, int):
                        bb.label_class = self.dataset.name_to_labels[
                            bb.label_class]

            if self._first_step == epoch or not use_reference:
                pcd_step = int(
                    np.ceil(pointcloud.shape[0] /
                            min(max_pts, pointcloud.shape[0])))
                pcd = pointcloud[::pcd_step, :3].cpu().detach()
                input_pcd.append(pcd)

        summary3d = {
            'input_pointcloud': {
                "vertex_positions":
                    input_pcd if self._first_step == epoch or not use_reference
                    else self._first_step
            },
            'objdet_prediction': {
                "bboxes": infer_bboxes_batch[:max_outputs],
                'label_to_names': self.dataset.get_label_to_names()
            }
        }
        if save_gt:
            summary3d['objdet_ground_truth'] = {
                "bboxes": inputs_batch.bbox_objs[:max_outputs],
                'label_to_names': self.dataset.get_label_to_names()
            }
        return summary3d

    def save_logs(self, writer, epoch):
        for key, val in self.losses.items():
            writer.add_scalar("train/" + key, np.mean(val), epoch)
        if (epoch % self.cfg.get("validation_freq", 1)) == 0:
            for key, val in self.valid_losses.items():
                writer.add_scalar("valid/" + key, np.mean(val), epoch)
        for stage in self.summary.keys():
            for key, summary_dict in self.summary[stage].items():
                label_to_names = summary_dict.pop('label_to_names', None)
                writer.add_3d('/'.join((stage, key)),
                              summary_dict,
                              epoch,
                              max_outputs=0,
                              label_to_names=label_to_names)

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        if self.rank == 0:
            make_dir(train_ckpt_dir)
        if self.distributed:
            dist.barrier()

        epoch = 0
        if ckpt_path is None:
            ckpt_path = latest_torch_ckpt(train_ckpt_dir)
            if ckpt_path is not None and is_resume:
                log.info('ckpt_path not given. Restore from the latest ckpt')
                epoch = int(re.findall(r'\d+', ckpt_path)[-1]) + 1
            else:
                log.info('Initializing from scratch.')
                return epoch

        if not exists(ckpt_path):
            raise FileNotFoundError(f' ckpt {ckpt_path} not found')

        log.info(f'Loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            log.info('Loading checkpoint optimizer_state_dict')
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler'):
            log.info('Loading checkpoint scheduler_state_dict')
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        return epoch

    def save_ckpt(self, epoch):
        path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(path_ckpt)
        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict()),
            # scheduler_state_dict=self.scheduler.state_dict()),
            join(path_ckpt, f'ckpt_{epoch:05d}.pth'))
        log.info(f'Epoch {epoch:3d}: save ckpt to {path_ckpt:s}')

    def save_config(self, writer):
        """Save experiment configuration with tensorboard summary."""
        if hasattr(self, 'cfg_tb'):
            writer.add_text("Description/Open3D-ML", self.cfg_tb['readme'], 0)
            writer.add_text("Description/Command line", self.cfg_tb['cmd_line'],
                            0)
            writer.add_text('Configuration/Dataset',
                            code2md(self.cfg_tb['dataset'], language='json'), 0)
            writer.add_text('Configuration/Model',
                            code2md(self.cfg_tb['model'], language='json'), 0)
            writer.add_text('Configuration/Pipeline',
                            code2md(self.cfg_tb['pipeline'], language='json'),
                            0)


PIPELINE._register_module(ObjectDetection, "torch")
