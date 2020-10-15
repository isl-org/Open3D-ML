import numpy as np
import logging
import sys
import warnings

from tqdm import tqdm
from datetime import datetime
from os.path import exists, join, isfile, dirname, abspath
from pathlib import Path

import tensorflow as tf

from .base_pipeline import BasePipeline
from ..modules.losses import SemSegLoss
from ..modules.metrics import SemSegMetric
from ..dataloaders import TFDataloader
from ...utils import make_dir, LogRecord, PIPELINE, get_runid, code2md

logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class SemanticSegmentation(BasePipeline):

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

    def run_inference(self, data):
        cfg = self.cfg
        model = self.model
        # model.eval()
        log.info("running inference")

        model.inference_begin(data)

        while True:
            inputs = model.inference_preprocess()
            results = model(inputs, training=False)
            if model.inference_end(results):
                break

        return model.inference_result

    def run_test(self):
        model = self.model
        dataset = self.dataset
        cfg = self.cfg

        self.load_ckpt(model.cfg.ckpt_path)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        log.info("Started testing")

        Metric = SemSegMetric(self, model, dataset)
        Loss = SemSegLoss(self, model, dataset)

        accs = []
        ious = []

        test_split = dataset.get_split('test')
        for idx in tqdm(range(len(test_split)), desc='test'):
            attr = test_split.get_attr(idx)
            data = test_split.get_data(idx)
            results = self.run_inference(data)
            scores, labels = Loss.filter_valid_label(results['predict_scores'],
                                                     data['label'])

            acc = Metric.acc(scores.numpy(), labels.numpy())
            iou = Metric.iou(scores.numpy(), labels.numpy())
            accs.append(acc)
            ious.append(iou)

            dataset.save_test_result(results, attr)

        accs = np.nanmean(np.array(accs), axis=0)
        ious = np.nanmean(np.array(ious), axis=0)

        log.info("Per class Accuracy : {}".format(accs[:-1]))
        log.info("Per class IOUs : {}".format(ious[:-1]))
        log.info("Overall Accuracy : {:.3f}".format(accs[-1]))
        log.info("Overall IOU : {:.3f}".format(ious[-1]))

    def run_train(self):
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
                                   use_cache=dataset.cfg.use_cache,
                                   steps_per_epoch=dataset.cfg.get(
                                       'steps_per_epoch_train', None))
        train_loader, len_train = train_split.get_loader(cfg.batch_size)

        valid_split = TFDataloader(dataset=dataset.get_split('validation'),
                                   model=model,
                                   use_cache=dataset.cfg.use_cache,
                                   steps_per_epoch=dataset.cfg.get(
                                       'steps_per_epoch_valid', None))
        valid_loader, len_val = valid_split.get_loader(cfg.val_batch_size)

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
        self.optimizer = model.get_optimizer(cfg)

        is_resume = model.cfg.get('is_resume', True)
        self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)
        for epoch in range(0, cfg.max_epoch + 1):
            log.info("=== EPOCH {}/{} ===".format(epoch, cfg.max_epoch))
            # --------------------- training
            self.accs = []
            self.ious = []
            self.losses = []
            step = 0

            for idx, inputs in enumerate(
                    tqdm(train_loader, total=len_train, desc='training')):
                with tf.GradientTape(persistent=True) as tape:
                    results = model(inputs, training=True)

                    loss, gt_labels, predict_scores = model.get_loss(
                        Loss, results, inputs)

                if len(predict_scores.shape) < 2:
                    continue

                if predict_scores.shape[0] == 0:
                    continue
                # params for deformable convolutions.
                scaled_params = []
                params = []
                for val in model.trainable_weights:
                    if 'deform' in val.name:
                        scaled_params.append(val)
                    else:
                        params.append(val)

                grads = tape.gradient(loss, params)
                scaled_grads = tape.gradient(loss, scaled_params)
                for i in range(len(scaled_grads)):
                    scaled_grads[i] *= 0.1

                norm = cfg.get('grad_clip_norm', 100.0)
                grads = [tf.clip_by_norm(g, norm) for g in grads]
                scaled_grads = [tf.clip_by_norm(g, norm) for g in scaled_grads]

                self.optimizer.apply_gradients(zip(grads, params))

                if len(scaled_grads) > 0:
                    self.optimizer.apply_gradients(
                        zip(scaled_grads, scaled_params))

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

            for idx, inputs in enumerate(
                    tqdm(valid_loader, total=len_val, desc='validation')):
                with tf.GradientTape() as tape:
                    results = model(inputs, training=False)
                    loss, gt_labels, predict_scores = model.get_loss(
                        Loss, results, inputs)

                if len(predict_scores.shape) < 2:
                    continue

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

        with warnings.catch_warnings():  # ignore Mean of empty slice.
            warnings.simplefilter('ignore', category=RuntimeWarning)
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

            for key, val in acc_dicts[-1].items():
                tf.summary.scalar("{}/ Overall".format(key), val, epoch)
            for key, val in iou_dicts[-1].items():
                tf.summary.scalar("{}/ Overall".format(key), val, epoch)

        # print(acc_dicts[-1])

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)

        if hasattr(self, 'optimizer'):
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                            optimizer=self.optimizer,
                                            net=self.model)
        else:
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=self.model)

        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  train_ckpt_dir,
                                                  max_to_keep=100)

        if ckpt_path is not None:
            self.ckpt.restore(ckpt_path).expect_partial()
            log.info("Restored from {}".format(ckpt_path))
        else:
            self.ckpt.restore(self.manager.latest_checkpoint)

            if self.manager.latest_checkpoint and is_resume:
                log.info("Restored from {}".format(
                    self.manager.latest_checkpoint))
            else:
                log.info("Initializing from scratch.")

    def save_ckpt(self, epoch):
        save_path = self.manager.save()
        log.info("Saved checkpoint at: {}".format(save_path))

    def save_config(self, writer):
        '''
        Save experiment configuration with tensorboard summary
        '''
        with writer.as_default():
            with tf.name_scope("Description"):
                tf.summary.text("Open3D-ML", self.cfg_tb['readme'], step=0)
                tf.summary.text("Command line", self.cfg_tb['cmd_line'], step=0)
            with tf.name_scope("Configuration"):
                tf.summary.text('Dataset',
                                code2md(self.cfg_tb['dataset'],
                                        language='json'),
                                step=0)
                tf.summary.text('Model',
                                code2md(self.cfg_tb['model'], language='json'),
                                step=0)
                tf.summary.text('Pipeline',
                                code2md(self.cfg_tb['pipeline'],
                                        language='json'),
                                step=0)


PIPELINE._register_module(SemanticSegmentation, "tf")
