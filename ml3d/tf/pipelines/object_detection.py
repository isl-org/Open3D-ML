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
from torch.utils.tensorboard import SummaryWriter
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
        log.info("running inference")

        inputs = tf.convert_to_tensor([data['point']], dtype=np.float32)

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

        log.info("Started testing")
        self.test_ious = []

        pred = []
        gt = []
        for i in tqdm(range(len(test_split)), desc='testing'):
            results = self.run_inference(test_split[i]['data'])

            pred.append(convert_data_eval(results[0], [40, 25]))
            gt.append(convert_data_eval(test_split[i]['data']['bboxes']))

        if cfg.get('test_compute_metric', True):
            ap = mAP(pred, gt, [0, 1, 2], [0, 1, 2], [0.5, 0.5, 0.7], similar_classes={0:4, 2:3})
            log.info("mAP BEV:")
            log.info("Pedestrian: {} (easy) {} (medium) {} (hard)".format(*ap[0,:,0]))
            log.info("Bicycle: {} (easy) {} (medium) {} (hard)".format(*ap[1,:,0]))
            log.info("Car: {} (easy) {} (medium) {} (hard)".format(*ap[2,:,0]))

            ap = mAP(pred, gt, [0, 1, 2], [0, 1, 2], [0.5, 0.5, 0.7], bev=False, similar_classes={0:4, 2:3})
            log.info("")
            log.info("mAP 3D:")
            log.info("Pedestrian: {} (easy) {} (medium) {} (hard)".format(*ap[0,:,0]))
            log.info("Bicycle: {} (easy) {} (medium) {} (hard)".format(*ap[1,:,0]))
            log.info("Car: {} (easy) {} (medium) {} (hard)".format(*ap[2,:,0]))

        #dataset.save_test_result(results, attr)


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
                                   transform=None,
                                   use_cache=False,
                                   get_batch_gen=model.get_batch_gen,
                                   shuffle=False)

        log.info("Started validation")

        self.valid_losses = {}
        self.valid_mAP = {}

        pred = []
        gt = []
        for i in tqdm(range(len(valid_loader)), desc='validation'):
            inputs = model.transform(valid_loader[i]['data'], None)
            results = model(inputs['point'])
            loss = model.loss(results, inputs)
            for l, v in loss.items():
                if not l in self.valid_losses:
                    self.valid_losses[l] = []
                self.valid_losses[l].append(v.numpy())

            # convert to bboxes for mAP evaluation
            boxes = model.inference_end(results, inputs)
            pred.append(convert_data_eval(boxes[0], [40, 25]))
            gt.append(convert_data_eval(valid_loader[i]['data']['bboxes']))
        
        sum_loss = 0
        desc = "validation - "
        for l, v in self.valid_losses.items():
            desc += " %s: %.03f" % (l, np.mean(v))
            sum_loss += np.mean(v)
        desc += " > loss: %.03f" % sum_loss

        log.info(desc)

        ap = mAP(pred, gt, [0, 1, 2], [0, 1, 2], [0.5, 0.5, 0.7], similar_classes={0:4, 2:3})
        log.info("mAP BEV:")
        log.info("Pedestrian: {} (easy) {} (medium) {} (hard)".format(*ap[0,:,0]))
        log.info("Bicycle: {} (easy) {} (medium) {} (hard)".format(*ap[1,:,0]))
        log.info("Car: {} (easy) {} (medium) {} (hard)".format(*ap[2,:,0]))

        ap = mAP(pred, gt, [0, 1, 2], [0, 1, 2], [0.5, 0.5, 0.7], bev=False, similar_classes={0:4, 2:3})
        log.info("")
        log.info("mAP 3D:")
        log.info("Pedestrian: {} (easy) {} (medium) {} (hard)".format(*ap[0,:,0]))
        log.info("Bicycle: {} (easy) {} (medium) {} (hard)".format(*ap[1,:,0]))
        log.info("Car: {} (easy) {} (medium) {} (hard)".format(*ap[2,:,0]))


    def run_train(self):
        model = self.model
        dataset = self.dataset

        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        train_dataset = dataset.get_split('training')
        train_loader = TFDataloader(dataset=train_dataset,
                                   model=model,
                                   use_cache=dataset.cfg.use_cache,
                                   steps_per_epoch=dataset.cfg.get(
                                      'steps_per_epoch_train', None))

        self.optimizer = model.get_optimizer(cfg.optimizer)

        is_resume = model.cfg.get('is_resume', True)
        start_ep = self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

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
        for epoch in range(start_ep, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{cfg.max_epoch:d} ===')
            self.losses = {}
            process_bar = tqdm(train_loader, desc='training')        
            for inputs in process_bar:
                with tf.GradientTape(persistent=True) as tape:
                    results = model(inputs['data']['point'])
                    loss = model.loss(results, inputs['data'])
                    loss_sum = sum(loss.values())

                grads = tape.gradient(loss, model.trainable_weights)
                
                norm = cfg.get('grad_clip_norm', -1)
                if model.cfg.get('grad_clip_norm', -1) > 0:
                    grads = [tf.clip_by_norm(g, norm) for g in grads]
                
                self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

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
            epoch = int(re.findall(r'\d+', ckpt_path)[-1])+1
        else:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()

            if self.manager.latest_checkpoint:
                log.info("Restored from {}".format(
                    self.manager.latest_checkpoint))
            else:
                log.info("Initializing from scratch.")
        return epoch


    def save_ckpt(self, epoch):
        save_path = self.manager.save()
        log.info("Saved checkpoint at: {}".format(save_path))


    def save_config(self, writer):
        '''
        Save experiment configuration with tensorboard summary
        '''
        writer.add_text("Description/Open3D-ML", self.cfg_tb['readme'], 0)
        writer.add_text("Description/Command line", self.cfg_tb['cmd_line'], 0)
        writer.add_text('Configuration/Dataset',
                        code2md(self.cfg_tb['dataset'], language='json'), 0)
        writer.add_text('Configuration/Model',
                        code2md(self.cfg_tb['model'], language='json'), 0)
        writer.add_text('Configuration/Pipeline',
                        code2md(self.cfg_tb['pipeline'], language='json'), 0)
                                

PIPELINE._register_module(ObjectDetection, "tf")
