import logging
import re
from datetime import datetime
from os.path import join
from pathlib import Path

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from open3d.visualization.tensorboard_plugin import summary as summary3d
from .base_pipeline import BasePipeline
from ..dataloaders import TFDataloader
from ...utils import make_dir, PIPELINE, get_runid, code2md
from ...datasets.utils import BEVBox3D, DataProcessing

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

        results = model(data, training=False)
        boxes = model.inference_end(results, data)

        return boxes

    def run_test(self):
        """Run test with test data split, computes mean average precision of the
        prediction results.
        """
        model = self.model
        dataset = self.dataset
        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        test_dataset = dataset.get_split('test')
        test_split = TFDataloader(dataset=test_dataset,
                                  model=model,
                                  use_cache=False)

        test_loader, len_test = test_split.get_loader(cfg.test_batch_size,
                                                      transform=False)

        self.load_ckpt(model.cfg.ckpt_path)

        if cfg.get('test_compute_metric', True):
            self.run_valid()

        log.info("Started testing")
        self.test_ious = []

        pred = []
        record_summary = 'test' in cfg.get('summary').get('record_for', [])
        process_bar = tqdm(test_loader, total=len_test, desc='testing')
        for data in process_bar:
            results = self.run_inference(data)
            pred.append(results[0])
            # Save only for the first batch
            if record_summary and 'test' not in self.summary:
                self.summary['test'] = self.get_3d_summary(results,
                                                           data,
                                                           0,
                                                           save_gt=False)

        # dataset.save_test_result(pred, attr)

    def run_valid(self, epoch=0):
        """Run validation with validation data split, computes mean average
        precision and the loss of the prediction results.

        Args:
            epoch (int): step for TensorBoard summary. Defaults to 0 if
                unspecified.
        """
        model = self.model
        dataset = self.dataset
        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        log_file_path = join(cfg.logs_dir, 'log_valid_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        valid_dataset = dataset.get_split('validation')
        valid_split = TFDataloader(dataset=valid_dataset,
                                   model=model,
                                   use_cache=False,
                                   steps_per_epoch=dataset.cfg.get(
                                       'steps_per_epoch_valid', None))

        valid_loader, len_valid = valid_split.get_loader(cfg.val_batch_size,
                                                         transform=False)

        record_summary = 'valid' in cfg.get('summary').get('record_for', [])
        log.info("Started validation")

        self.valid_losses = {}

        pred = []
        gt = []

        process_bar = tqdm(valid_loader, total=len_valid, desc='validation')
        for i, data in enumerate(process_bar):
            results = model(data, training=False)
            loss = model.loss(results, data, training=False)
            for l, v in loss.items():
                if l not in self.valid_losses:
                    self.valid_losses[l] = []
                self.valid_losses[l].append(v.numpy())

            # convert to bboxes for mAP evaluation
            boxes = model.inference_end(results, data)
            pred.extend([BEVBox3D.to_dicts(b) for b in boxes])
            gt.extend([
                BEVBox3D.to_dicts(valid_split[i * cfg.val_batch_size +
                                              bi]["data"]["bbox_objs"])
                for bi in range(cfg.val_batch_size)
            ])
            # Save only for the first batch
            if record_summary and 'valid' not in self.summary:
                self.summary['valid'] = self.get_3d_summary(boxes,
                                                            data,
                                                            epoch,
                                                            results=results)

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
        """Run training with train data split."""
        model = self.model
        dataset = self.dataset

        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        train_dataset = dataset.get_split('training')
        train_split = TFDataloader(dataset=train_dataset,
                                   model=model,
                                   use_cache=dataset.cfg.use_cache,
                                   steps_per_epoch=dataset.cfg.get(
                                       'steps_per_epoch_train', None))
        train_loader, len_train = train_split.get_loader(cfg.batch_size,
                                                         transform=False)

        self.optimizer = model.get_optimizer(cfg.optimizer)

        is_resume = model.cfg.get('is_resume', True)
        start_ep = self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

        dataset_name = dataset.name if dataset is not None else ''
        tensorboard_dir = join(
            self.cfg.train_sum_dir,
            model.__class__.__name__ + '_' + dataset_name + '_tf')
        runid = get_runid(tensorboard_dir)
        self.tensorboard_dir = join(self.cfg.train_sum_dir,
                                    runid + '_' + Path(tensorboard_dir).name)

        writer = tf.summary.create_file_writer(self.tensorboard_dir)
        self.save_config(writer)
        log.info("Writing summary in {}.".format(self.tensorboard_dir))
        record_summary = 'train' in cfg.get('summary').get('record_for', [])

        log.info("Started training")
        for epoch in range(start_ep, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            self.losses = {}
            process_bar = tqdm(train_loader, total=len_train, desc='training')
            for data in process_bar:
                with tf.GradientTape(persistent=True) as tape:
                    results = model(data)
                    loss = model.loss(results, data)
                    loss_sum = tf.add_n(loss.values())

                grads = tape.gradient(loss_sum, model.trainable_weights)
                for (grad, var) in zip(grads, model.trainable_weights):
                    if grad is None:
                        print(var.shape, var.name, grad)

                norm = cfg.get('grad_clip_norm', -1)
                if model.cfg.get('grad_clip_norm', -1) > 0:
                    grads = [tf.clip_by_norm(g, norm) for g in grads]

                self.optimizer.apply_gradients(
                    zip(grads, model.trainable_weights))

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
                    self.losses[l].append(v.numpy())
                    desc += " %s: %.03f" % (l, v.numpy())
                desc += " > loss: %.03f" % loss_sum.numpy()
                process_bar.set_description(desc)
                process_bar.refresh()

            # --------------------- validation
            self.run_valid()

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
            inputs_batch: Batch of ground truth boxes and pointclouds.
                PointPillars: (Tuple (points(N, 4), bboxes(Nb, 7), labels(Nb),
                    calib(B, 2, 4, 4), points_batch_lengths(B,),
                    bboxes_batch_lengths(B,)))
                PointRCNN: (Tuple (points(B, N, 3), bboxes(B, Nb, 7), labels(B,
                    Nb), calib(B, 2, 4, 4)))
            epoch (int): step
            results (tf.FloatTensor): Model output (only required for RPN
                stage of PointRCNN)
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
        input_pcd = []
        gt_bboxes = []
        if self.model.cfg['name'] == 'PointPillars':
            max_outputs = min(cfg.get('max_outputs', 1), len(inputs_batch[4]))
            (pointclouds, bboxes, labels, calib, points_batch_lengths,
             bboxes_batch_lengths) = inputs_batch
            gt_bboxes = [[] for _ in range(max_outputs)]
            pt_start_idx, box_start_idx = 0, 0
            for bidx in range(max_outputs):
                if self._first_step == epoch or not use_reference:
                    ptblen = points_batch_lengths[bidx]
                    pcd_step = int(np.ceil(ptblen / min(max_pts, ptblen)))
                    pcd = pointclouds[pt_start_idx:pt_start_idx +
                                      ptblen:pcd_step, :3]
                    input_pcd.append(pcd)
                    pt_start_idx += ptblen
                world_cam, cam_img = calib[bidx].numpy()
                boxblen = points_batch_lengths[bidx]
                for bbox, label in zip(
                        bboxes[box_start_idx:box_start_idx + boxblen],
                        labels[box_start_idx:box_start_idx + boxblen]):
                    dim = tf.gather(bbox, [3, 5, 4])
                    pos = bbox[:3] + [0, 0, dim[1] / 2]
                    yaw = bbox[-1]
                    gt_bboxes[bidx].append(
                        BEVBox3D(pos, dim, yaw, label, 1, world_cam, cam_img))

        elif self.model.cfg['name'] == 'PointRCNN':
            pointclouds, bboxes, labels, calib = inputs_batch
            max_outputs = min(cfg.get('max_outputs', 1), pointclouds.shape[0])
            gt_bboxes = [[] for _ in range(max_outputs)]
            pcd_step = int(
                np.ceil(pointclouds.shape[1] /
                        min(max_pts, pointclouds.shape[1])))
            if self.model.mode == 'RPN':
                for bidx in range(max_outputs):
                    if self._first_step == epoch or not use_reference:
                        pcd = pointclouds[bidx, ::pcd_step, :3]
                        input_pcd.append(pcd)
                rpn_scores_norm = tf.sigmoid(results['cls'])
                seg_mask = tf.cast((rpn_scores_norm > self.model.score_thres),
                                   tf.float32)
                cls_score = [tf.stop_gradient(ten) for ten in seg_mask]
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
                return summary3d

            for bidx in range(max_outputs):
                if self._first_step == epoch or not use_reference:
                    pcd = pointclouds[bidx, ::pcd_step, :3]
                    input_pcd.append(pcd)
                world_cam, cam_img = calib[bidx].numpy()
                bboxes = bboxes.numpy()
                labels = labels.numpy()
                for bbox, label in zip(bboxes[bidx], labels[bidx]):
                    pos = bbox[:3]
                    dim = tf.gather(bbox, [4, 3, 5])
                    # transform into world space
                    pos = DataProcessing.cam2world(pos.reshape((1, -1)),
                                                   world_cam).flatten()
                    pos = pos + [0, 0, dim[1] / 2]
                    yaw = bbox[-1]
                    gt_bboxes[bidx].append(
                        BEVBox3D(pos, dim, yaw, label, 1, world_cam, cam_img))
        else:
            raise NotImplementedError(
                f"Saving 3D summary for the model {self.model.cfg['name']}"
                " is not implemented.")

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
                "bboxes": gt_bboxes,
                'label_to_names': self.dataset.get_label_to_names()
            }
        return summary3d

    def save_logs(self, writer, epoch):
        with writer.as_default():
            for key, val in self.losses.items():
                tf.summary.scalar("train/" + key, np.mean(val), epoch)

            for key, val in self.valid_losses.items():
                tf.summary.scalar("valid/" + key, np.mean(val), epoch)
            for stage in self.summary.keys():
                for key, summary_dict in self.summary[stage].items():
                    label_to_names = summary_dict.pop('label_to_names', None)
                    summary3d.add_3d('/'.join((stage, key)),
                                     summary_dict,
                                     epoch,
                                     max_outputs=0,
                                     label_to_names=label_to_names,
                                     logdir=self.tensorboard_dir)

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)

        if hasattr(self, 'optimizer'):
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                            optimizer=self.optimizer,
                                            model=self.model)
        else:
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                            model=self.model)

        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  train_ckpt_dir,
                                                  max_to_keep=100)

        epoch = 0
        if ckpt_path is not None:
            self.ckpt.restore(ckpt_path).expect_partial()
            log.info("Restored from {}".format(ckpt_path))
        else:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()

            if self.manager.latest_checkpoint:
                log.info("Restored from {}".format(
                    self.manager.latest_checkpoint))
                epoch = int(
                    re.findall(r'\d+', self.manager.latest_checkpoint)[-1]) - 1
                epoch = epoch * self.cfg.save_ckpt_freq + 1
            else:
                log.info("Initializing from scratch.")
        return epoch

    def save_ckpt(self, epoch):
        save_path = self.manager.save()
        log.info("Saved checkpoint at: {}".format(save_path))

    def save_config(self, writer):
        """Save experiment configuration with tensorboard summary."""
        if hasattr(self, 'cfg_tb'):
            with writer.as_default():
                with tf.name_scope("Description"):
                    tf.summary.text("Open3D-ML", self.cfg_tb['readme'], step=0)
                    tf.summary.text("Command line",
                                    self.cfg_tb['cmd_line'],
                                    step=0)
                with tf.name_scope("Configuration"):
                    tf.summary.text('Dataset',
                                    code2md(self.cfg_tb['dataset'],
                                            language='json'),
                                    step=0)
                    tf.summary.text('Model',
                                    code2md(self.cfg_tb['model'],
                                            language='json'),
                                    step=0)
                    tf.summary.text('Pipeline',
                                    code2md(self.cfg_tb['pipeline'],
                                            language='json'),
                                    step=0)


PIPELINE._register_module(ObjectDetection, "tf")
