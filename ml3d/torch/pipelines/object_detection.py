import torch
import logging
from tqdm import tqdm
import numpy as np
import re

from datetime import datetime

from os.path import exists, join
from torch.utils.data import DataLoader
from pathlib import Path

from .base_pipeline import BasePipeline
from ..dataloaders import TorchDataloader, ConcatBatcher
from torch.utils.tensorboard import SummaryWriter
from ..utils import latest_torch_ckpt
from ...utils import make_dir, PIPELINE, LogRecord, get_runid, code2md
from ...datasets.utils import BEVBox3D
from ...vis import BoundingBox3D
from ...metrics.mAP import mAP

# import torch.cuda.profiler as profiler
# import pyprof
# pyprof.init(enable_function_stack=True)

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
        """
        Run inference on given data.

        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """
        model = self.model

        model.eval()

        data.to(self.device)

        with torch.no_grad():
            results = model(data)
            boxes = model.inference_end(results, data)

        return boxes

    def run_test(self):
        """
        Run test with test data split, computes mean average precision of the
        prediction results.
        """
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg

        model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

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
        )

        self.load_ckpt(model.cfg.ckpt_path)

        if cfg.get('test_compute_metric', True):
            self.run_valid()

        log.info("Started testing")
        self.test_ious = []

        pred = []
        with torch.no_grad():
            for data in tqdm(test_loader, desc='testing'):
                results = self.run_inference(data)
                pred.append(results[0])

        # dataset.save_test_result(results, attr)

    def run_valid(self):
        """
        Run validation with validation data split, computes mean average
        precision and the loss of the prediction results.
        """
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg

        model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_valid_' + timestamp + '.txt')
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
        valid_loader = DataLoader(
            valid_split,
            batch_size=cfg.val_batch_size,
            num_workers=cfg.get('num_workers', 4),
            pin_memory=cfg.get('pin_memory', True),
            collate_fn=batcher.collate_fn,
        )

        log.info("Started validation")

        self.valid_losses = {}
        self.val_anchor_vis = []

        pred = []
        gt = []
        no_bboxes = 0
        with torch.no_grad():
            process_bar = tqdm(valid_loader, desc='validation')
            for data in process_bar:
                if any([bbox.numel() == 0 for bbox in data.bboxes]):
                    no_bboxes += 1
                    continue
                data.to(device)
                results = model(data)
                loss = model.loss(results, data)
                for l, v in loss.items():
                    if l not in self.valid_losses:
                        self.valid_losses[l] = []
                    self.valid_losses[l].append(v.cpu().item())

                # convert to bboxes for mAP evaluation
                boxes = model.inference_end(results, data)
                pred.extend([BEVBox3D.to_dicts(b) for b in boxes])
                gt.extend([BEVBox3D.to_dicts(b) for b in data.bbox_objs])
                # Record bboxes for the last iteration
                if process_bar.n == process_bar.total - 1:
                    self.val_anchor_vis.append(self.get_visual(boxes, data))

            if no_bboxes > 0:
                log.warning("No bounding box labels in " +
                            f"{no_bboxes}/{len(process_bar)} cases.")
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
        """
        Run training with train data split.
        """
        model = self.model
        device = self.device
        dataset = self.dataset

        cfg = self.cfg

        log.info("DEVICE : {}".format(device))
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
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
        train_loader = DataLoader(
            train_split,
            batch_size=cfg.batch_size,
            num_workers=cfg.get('num_workers', 4),
            pin_memory=cfg.get('pin_memory', True),
            collate_fn=batcher.collate_fn,
        )

        self.optimizer, self.scheduler = model.get_optimizer(cfg.optimizer)

        is_resume = model.cfg.get('is_resume', True)
        start_ep = self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

        dataset_name = dataset.name if dataset is not None else ''
        if hasattr(dataset, 'portion'):
            dataset_name += "_" + dataset.portion
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
        # with torch.autograd.profiler.emit_nvtx(enabled=False):
        for epoch in range(start_ep, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            model.train()

            self.losses = {}
            self.train_anchor_vis = []
            no_bboxes = 0
            process_bar = tqdm(train_loader, desc='training')
            for data in process_bar:
                if any([bbox.numel() == 0 for bbox in data.bboxes]):
                    no_bboxes += 1
                    continue
                data.to(device)
                results = model(data)
                loss = model.loss(results, data)
                loss_sum = sum(loss.values())
                # Record visualization for the last iteration
                # if process_bar.n == process_bar.total - 1:
                #     boxes = model.inference_end(results, data)
                #     self.train_anchor_vis.append(self.get_visual(boxes, data))

                self.optimizer.zero_grad()
                loss_sum.backward()
                if model.cfg.get('grad_clip_norm', -1) > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(),
                                                    model.cfg.grad_clip_norm)
                self.optimizer.step()
                desc = "training - "
                for l, v in loss.items():
                    if l not in self.losses:
                        self.losses[l] = []
                    self.losses[l].append(v.cpu().item())
                    desc += " %s: %.03f" % (l, v.cpu().item())
                desc += " > loss: %.03f" % loss_sum.cpu().item()
                process_bar.set_description(desc)
                process_bar.refresh()

                if no_bboxes > 0:
                    log.warning("No bounding box labels in " +
                                f"{no_bboxes}/{len(process_bar)} cases.")
                # self.scheduler.step()

                # --------------------- validation
                self.run_valid()

                self.save_logs(writer, epoch)

                if epoch % cfg.save_ckpt_freq == 0:
                    self.save_ckpt(epoch)

    def get_visual(self, infer_bboxes_batch, inputs_batch):
        """
        inputs.bbox_objs: List[List[Object3D]]
        infer_bboxes:
        """
        points = []
        colors = []
        faces = []
        input_pcd = []
        max_pcd_pts = 10000
        max_pts = 0
        max_faces = 0
        for infer_bboxes, gt_bboxes, pointcloud in zip(infer_bboxes_batch,
                                                       inputs_batch.bbox_objs,
                                                       inputs_batch.point):
            gt_points, gt_colors, gt_faces = BoundingBox3D.create_trimesh(
                gt_bboxes, lut=self.dataset.label_lut)
            pred_points, pred_colors, pred_faces = BoundingBox3D.create_trimesh(
                infer_bboxes)
            pred_faces += gt_points.shape[0]
            max_pts = max(max_pts, gt_points.shape[0] + pred_points.shape[0])
            max_faces = max(max_faces, gt_faces.shape[0] + pred_faces.shape[0])
            points.append(np.vstack((gt_points, pred_points)))
            colors.append(np.vstack((gt_colors, pred_colors)))
            faces.append(np.vstack((gt_faces, pred_faces)))
            pcd_subsample = np.linspace(0,
                                        pointcloud.shape[0] - 1,
                                        num=max_pcd_pts,
                                        dtype=int)
            input_pcd.append(pointcloud[pcd_subsample, :].cpu().numpy())

        points = np.stack(
            [np.pad(p, ((0, max_pts - p.shape[0]), (0, 0))) for p in points])
        colors = np.stack(
            [np.pad(c, ((0, max_pts - c.shape[0]), (0, 0))) for c in colors])
        faces = np.stack(
            [np.pad(f, ((0, max_faces - f.shape[0]), (0, 0))) for f in faces])
        input_pcd = np.stack(input_pcd)

        return {
            'points': points,
            'colors': colors,
            'faces': faces,
            'input_pcd': input_pcd
        }

    def save_logs(self, writer, epoch):
        anchor_config = {
            "material": {
                "cls": "MeshBasicMaterial",
                "wireframe": True
            },
            "camera": {
                "cls": "OrthographicCamera",
                "near": 0.25,
                "far": 3.5
            }
        }
        pcd_config = {
            "camera": {
                "cls": "OrthographicCamera",
                "near": 0.25,
                "far": 3.5
            }
        }
        for key, val in self.losses.items():
            writer.add_scalar("train/" + key, np.mean(val), epoch)
        for key, mesh in enumerate(self.train_anchor_vis):
            writer.add_mesh(f"train/anchors/{key}",
                            vertices=torch.from_numpy(mesh['points']),
                            colors=torch.from_numpy(mesh['colors']),
                            faces=torch.from_numpy(mesh['faces']),
                            config_dict=anchor_config,
                            global_step=epoch)
            writer.add_mesh(f"train/pointcloud/{key}",
                            vertices=torch.from_numpy(mesh['input_pcd']),
                            config_dict=pcd_config,
                            global_step=epoch)

        for key, val in self.valid_losses.items():
            writer.add_scalar("valid/" + key, np.mean(val), epoch)
        for key, mesh in enumerate(self.val_anchor_vis):
            writer.add_mesh(f"valid/anchors/{key}",
                            vertices=torch.from_numpy(mesh['points']),
                            colors=torch.from_numpy(mesh['colors']),
                            faces=torch.from_numpy(mesh['faces']),
                            config_dict=anchor_config,
                            global_step=epoch)
            writer.add_mesh(f"train/pointcloud/{key}",
                            vertices=torch.from_numpy(mesh['input_pcd']),
                            config_dict=pcd_config,
                            global_step=epoch)

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)

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

        return ckpt['epoch'] + 1

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
        '''
        Save experiment configuration with tensorboard summary
        '''
        if 'about' in self.cfg_tb:
            writer.add_text("About", self.cfg_tb['about'], 0)
        writer.add_text("Description/Open3D-ML", self.cfg_tb['readme'], 0)
        writer.add_text("Description/Command line", self.cfg_tb['cmd_line'], 0)
        writer.add_text('Configuration/Dataset',
                        code2md(self.cfg_tb['dataset'], language='json'), 0)
        writer.add_text('Configuration/Model',
                        code2md(self.cfg_tb['model'], language='json'), 0)
        writer.add_text('Configuration/Pipeline',
                        code2md(self.cfg_tb['pipeline'], language='json'), 0)


PIPELINE._register_module(ObjectDetection, "torch")
