#coding: future_fstrings
import numpy as np
import logging
import sys

from datetime import datetime
from tqdm import tqdm

from os.path import exists, join, isfile, dirname, abspath
import tensorflow as tf
import yaml


from ..modules.losses import SemSegLoss
from ..modules.metrics import SemSegMetric
from ..dataloaders import TFDataloader
from ...utils import make_dir, LogRecord



logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class SemanticSegmentation():
    def __init__(self, model, dataset, cfg):
        self.model = model
        self.dataset = dataset
        self.cfg = cfg

        make_dir(cfg.main_log_dir)
        cfg.logs_dir = join(cfg.main_log_dir, cfg.model_name)
        make_dir(cfg.logs_dir)

        # dataset.cfg.num_points = model.cfg.num_points

    def run_inference(self, points, device):
        # TODO
        pass

    def run_test(self, device):
        # TODO
        pass

    def run_train(self):
        model = self.model
        dataset = self.dataset

        cfg = self.cfg

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            cfg.adam_lr, decay_steps=100000, decay_rate=cfg.scheduler_gamma)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        Loss = SemSegLoss(self, model, dataset)
        Metric = SemSegMetric(self, model, dataset)

        train_split = TFDataloader(dataset=dataset.get_split('training'), 
                                 model = model)
        train_loader = train_split.get_loader().batch(cfg.batch_size)


        for epoch in range(0, cfg.max_epoch + 1): 
            print(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            self.accs = []
            self.ious = []
            self.losses = []
            step = 0

            #for inputs in train_loader:
            for idx, inputs in enumerate(tqdm(train_loader)):
                with tf.GradientTape() as tape:
                    results = model(inputs, training=True)  
                    loss, gt_labels, predict_scores = model.loss(
                        Loss, results, inputs)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                acc = Metric.acc(predict_scores, gt_labels)
                iou = Metric.iou(predict_scores, gt_labels)
                self.losses.append(loss.numpy())
                self.accs.append(acc)
                self.ious.append(iou)
                step = step + 1



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
        log.info(f"acc train: {iou_dicts[-1]['Training IoU']:.3f} "
                 f" eval: {iou_dicts[-1]['Validation IoU']:.3f}")

        # print(acc_dicts[-1])

    def load_ckpt(self, ckpt_path, is_train=True):
        # TODO
        pass


    def save_ckpt(self, path_ckpt, epoch):
        # TODO
        pass