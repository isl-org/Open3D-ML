#coding: future_fstrings
import numpy as np
import logging
import sys
from tqdm import tqdm
from datetime import datetime
from os.path import exists, join, isfile, dirname, abspath

import tensorflow as tf

from .base_pipeline import BasePipeline
from ..modules.losses import SemSegLoss
from ..modules.metrics import SemSegMetric
from ..dataloaders import TFDataloader
from ...utils import make_dir, LogRecord, PIPELINE

logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

class SemanticSegmentation(BasePipeline):
    def __init__(self, 
                model=None, 
                dataset=None, 
                cfg=None,  
                device=None,
                **kwargs):
    
        self.default_cfg_name = "semantic_segmentation.yml"

        super().__init__(model=model, 
                        dataset=dataset, 
                        cfg=cfg,  
                        device=device,
                        **kwargs)

<<<<<<< HEAD
=======
        make_dir(cfg.main_log_dir)
        cfg.logs_dir = join(cfg.main_log_dir, 
                    model.__class__.__name__ + '_TF')
        make_dir(cfg.logs_dir)
>>>>>>> master

        # tf.config.gpu.set_per_process_memory_growth(True)


        # dataset.cfg.num_points = model.cfg.num_points

    def run_inference(self, points, device):
        # TODO
        pass

    def run_test(self, device):
        # TODO
        pass

    def run_train(self, **kwargs):
        model = self.model
        dataset = self.dataset

        cfg = self.cfg

        log.info(model)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        

        Loss = SemSegLoss(self, model, dataset)
        Metric = SemSegMetric(self, model, dataset)

        train_split = TFDataloader(dataset=dataset.get_split('training'),
                                   model=model,
                                   use_cache=dataset.cfg.use_cache)
        train_loader = train_split.get_loader(cfg.batch_size)

        valid_split = TFDataloader(dataset=dataset.get_split('validation'),
                                   model=model,
                                   use_cache=dataset.cfg.use_cache)
        valid_loader = valid_split.get_loader(cfg.val_batch_size)

        writer = tf.summary.create_file_writer(
                    join(cfg.logs_dir, cfg.train_sum_dir))


        self.optimizer = model.get_optimizer(cfg)
        self.load_ckpt()
        for epoch in range(0, cfg.max_epoch + 1):
            log.info("=== EPOCH {}/{} ===".format(epoch, cfg.max_epoch))
            # --------------------- training
            self.accs = []
            self.ious = []
            self.losses = []
            step = 0

            for idx, inputs in enumerate(tqdm(train_loader, desc='training')):
                with tf.GradientTape() as tape:
                    results = model(inputs, training=True)
                    loss, gt_labels, predict_scores = model.get_loss(
                        Loss, results, inputs)

                grads = tape.gradient(loss, model.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

                acc = Metric.acc(predict_scores, gt_labels)
                iou = Metric.iou(predict_scores, gt_labels)

                self.losses.append(loss.numpy())
                self.accs.append(acc)
                self.ious.append(iou)
                step = step + 1



            # --------------------- validation
            self.valid_accs = []
            self.valid_ious = []
            self.valid_losses = []
            step = 0

            for idx, inputs in enumerate(tqdm(valid_loader, desc='validation')):
                with tf.GradientTape() as tape:
                    results = model(inputs, training=False)
                    loss, gt_labels, predict_scores = model.get_loss(
                        Loss, results, inputs)

                acc = Metric.acc(predict_scores, gt_labels)
                iou = Metric.iou(predict_scores, gt_labels)

                self.valid_losses.append(loss.numpy())
                self.valid_accs.append(acc)
                self.valid_ious.append(iou)
                step = step + 1
   

            self.save_logs(writer, epoch)

            if epoch % cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch)

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

        log.info(f"loss train: {loss_dict['Training loss']:.3f} "
                 f" eval: {loss_dict['Validation loss']:.3f}")
        log.info(f"acc train: {acc_dicts[-1]['Training accuracy']:.3f} "
                 f" eval: {acc_dicts[-1]['Validation accuracy']:.3f}")
        log.info(f"iou train: {iou_dicts[-1]['Training IoU']:.3f} "
                 f" eval: {iou_dicts[-1]['Validation IoU']:.3f}")

        # send results to tensorboard
        with writer.as_default():
            for key, val in loss_dict.items():
                tf.summary.scalar(key, val, epoch)
            for i in range(self.model.cfg.num_classes):
                for key, val in acc_dicts[i].items():
                    tf.summary.scalar("{}/{}".format(key,i), val, epoch)
                for key, val in iou_dicts[i].items():
                    tf.summary.scalar("{}/{}".format(key,i), val, epoch)

            for key, val in acc_dicts[-1].items():
                tf.summary.scalar("{}/ Overall".format(key), val, epoch)
            for key, val in iou_dicts[-1].items():
                tf.summary.scalar("{}/ Overall".format(key), val, epoch)

        # print(acc_dicts[-1])

    def load_ckpt(self):
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), 
                            optimizer=self.optimizer,
                            net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, 
                train_ckpt_dir, max_to_keep=3)


        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        #if exists(self.model.cfg.ckpt_path):
        #    self.model.load_weights(self.model.cfg.ckpt_path)
        #    log.info("Loading checkpoint {}".format(self.model.cfg.ckpt_path))


    def save_ckpt(self, epoch):
        save_path = self.manager.save()
        log.info("Saved checkpoint at: {}".format( save_path))
     
<<<<<<< HEAD

PIPELINE._register_module(SemanticSegmentation, 'tf')
=======
PIPELINE._register_module(SemanticSegmentation, "tf")
>>>>>>> master
