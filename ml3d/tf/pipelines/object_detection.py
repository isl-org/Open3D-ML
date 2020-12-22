import tensorflow as tf
import logging
import numpy as np
from tqdm import tqdm
import re

from datetime import datetime

from os.path import exists, join
from pathlib import Path

from .base_pipeline import BasePipeline
from ..dataloaders import TFDataloader
from ...utils import make_dir, PIPELINE, LogRecord, get_runid, code2md

from ...metrics.mAP import mAP, convert_data_eval

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
        Run inference on a given data.

        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """
        model = self.model

        inputs = tf.convert_to_tensor(data['point'], dtype=np.float32)

        results = model(inputs, training=False)
        boxes = model.inference_end(results, data)

        return boxes

    def run_test(self):
        """
        Run test with test data split, computes mean average precision of the prediction results.
        """
        model = self.model
        dataset = self.dataset
        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        test_dataset = dataset.get_split('test')
        test_split = TFDataloader(dataset=test_dataset,
                                  preprocess=model.preprocess,
                                  transform=None,
                                  use_cache=False,
                                  get_batch_gen=model.get_batch_gen,
                                  shuffle=False)

        self.load_ckpt(model.cfg.ckpt_path)

        if cfg.get('test_compute_metric', True):
            self.run_valid()

        log.info("Started testing")
        self.test_ious = []

        pred = []
        for i in tqdm(range(len(test_split)), desc='testing'):
            results = self.run_inference(test_split[i]['data'])
            pred.append(convert_data_eval(results[0], [40, 25]))

        #dataset.save_test_result(pred, attr)

    def run_valid(self):
        model = self.model
        dataset = self.dataset
        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_valid_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        valid_dataset = dataset.get_split('validation')
        valid_loader = TFDataloader(dataset=valid_dataset,
                                    preprocess=model.preprocess,
                                    transform=model.transform,
                                    use_cache=False,
                                    get_batch_gen=model.get_batch_gen,
                                    shuffle=False)

        log.info("Started validation")

        self.valid_losses = {}
        self.valid_mAP = {}

        pred = []
        gt = []
        for i in tqdm(range(len(valid_loader)), desc='validation'):
            data = valid_loader[i]['data']
            results = model(data['point'], training=False)
            loss = model.loss(results, data)
            for l, v in loss.items():
                if not l in self.valid_losses:
                    self.valid_losses[l] = []
                self.valid_losses[l].append(v.numpy())

            # convert to bboxes for mAP evaluation
            boxes = model.inference_end(results, data)
            pred.append(convert_data_eval(boxes[0], [40, 25]))
            gt.append(convert_data_eval(data['bbox_objs']))

        sum_loss = 0
        desc = "validation - "
        for l, v in self.valid_losses.items():
            desc += " %s: %.03f" % (l, np.mean(v))
            sum_loss += np.mean(v)
        desc += " > loss: %.03f" % sum_loss

        log.info(desc)

        ap = mAP(pred,
                 gt, [0, 1, 2], [0, 1, 2], [0.5, 0.5, 0.7],
                 similar_classes={
                     0: 4,
                     2: 3
                 })
        log.info("mAP BEV:")
        log.info(
            "Pedestrian:   {} (easy) {} (medium) {} (hard)".format(*ap[0, :,
                                                                       0]))
        log.info(
            "Bicycle:      {} (easy) {} (medium) {} (hard)".format(*ap[1, :,
                                                                       0]))
        log.info(
            "Car:          {} (easy) {} (medium) {} (hard)".format(*ap[2, :,
                                                                       0]))
        log.info("Overall:      {}".format(np.mean(ap[:, 2])))
        self.valid_losses["mAP BEV"] = np.mean(ap[:, 2])

        ap = mAP(pred,
                 gt, [0, 1, 2], [0, 1, 2], [0.5, 0.5, 0.7],
                 bev=False,
                 similar_classes={
                     0: 4,
                     2: 3
                 })
        log.info("")
        log.info("mAP 3D:")
        log.info(
            "Pedestrian:   {} (easy) {} (medium) {} (hard)".format(*ap[0, :,
                                                                       0]))
        log.info(
            "Bicycle:      {} (easy) {} (medium) {} (hard)".format(*ap[1, :,
                                                                       0]))
        log.info(
            "Car:          {} (easy) {} (medium) {} (hard)".format(*ap[2, :,
                                                                       0]))
        log.info("Overall:      {}".format(np.mean(ap[:, 2])))
        self.valid_losses["mAP 3D"] = np.mean(ap[:, 2])

    def run_train(self):
        model = self.model
        dataset = self.dataset

        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        train_dataset = dataset.get_split('training')
        train_split = TFDataloader(dataset=train_dataset,
                                    model=model,
                                    use_cache=dataset.cfg.use_cache,
                                    steps_per_epoch=dataset.cfg.get(
                                        'steps_per_epoch_train', None))
        train_loader, len_train = train_split.get_loader(cfg.batch_size, transform=False)

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

        log.info("Started training")
        for epoch in range(start_ep, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            self.losses = {}
            process_bar = tqdm(train_loader, total=len_train, desc='training')
            for data in process_bar:
                inputs, cnts_pts, cnts_lbs = data[:-2], data[-2], data[-1]
                with tf.GradientTape(persistent=True) as tape:
                    results = model(inputs[0], cnts=cnts_pts)
                    loss = model.loss(results, inputs, cnts=cnts_lbs)
                    loss_sum = tf.add_n(loss.values())

                grads = tape.gradient(loss_sum, model.trainable_weights)

                norm = cfg.get('grad_clip_norm', -1)
                if model.cfg.get('grad_clip_norm', -1) > 0:
                    grads = [tf.clip_by_norm(g, norm) for g in grads]

                self.optimizer.apply_gradients(
                    zip(grads, model.trainable_weights))

                desc = "training - "
                for l, v in loss.items():
                    if not l in self.losses:
                        self.losses[l] = []
                    self.losses[l].append(v.numpy())
                    desc += " %s: %.03f" % (l, v.numpy())
                desc += " > loss: %.03f" % loss_sum.numpy()
                process_bar.set_description(desc)
                process_bar.refresh()

            #self.scheduler.step()

            # --------------------- validation
            self.run_valid()

            self.save_logs(writer, epoch)

            if epoch % cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch)

    def save_logs(self, writer, epoch):
        with writer.as_default():
            for key, val in self.losses.items():
                tf.summary.scalar("train/" + key, np.mean(val), epoch)

            for key, val in self.valid_losses.items():
                tf.summary.scalar("valid/" + key, np.mean(val), epoch)

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


PIPELINE._register_module(ObjectDetection, "tf")
