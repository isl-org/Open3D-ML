import logging
from datetime import datetime
from os.path import join
from pathlib import Path

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from open3d.visualization.tensorboard_plugin import summary as summary3d
from .base_pipeline import BasePipeline
from ..modules.losses import SemSegLoss
from ..modules.metrics import SemSegMetric
from ..dataloaders import TFDataloader
from ...utils import make_dir, PIPELINE, get_runid, code2md

log = logging.getLogger(__name__)


class SemanticSegmentation(BasePipeline):
    """This class allows you to perform semantic segmentation for both training
    and inference using the TensorFlow framework. This pipeline has multiple
    stages: Pre-processing, loading dataset, testing, and inference or training.

    **Example:**
        This example loads the Semantic Segmentation and performs a training
        using the SemanticKITTI dataset.

        .. code::

            import tensorflow as tf
            from .base_pipeline import BasePipeline

            Mydataset = TFDataloader(dataset=tf.dataset.get_split('training')
            MyModel = SemanticSegmentation(self,model,dataset=Mydataset,
                name='SemanticSegmentation',
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
            dataset: The 3D ML dataset class. You can use the base dataset,
                sample datasets, or a custom dataset.
            model: The model to be used for building the pipeline.
            name: The name of the current training.
            batch_size: The batch size to be used for training.
            val_batch_size: The batch size to be used for validation.
            test_batch_size: The batch size to be used for testing.
            max_epoch: The maximum size of the epoch to be used for training.
            leanring_rate: The hyperparameter that controls the weights during
                training. Also, known as step size.
            lr_decays: The learning rate decay for the training.
            save_ckpt_freq: The frequency in which the checkpoint should be
                saved.
            adam_lr: The leanring rate to be applied for Adam optimization.
            scheduler_gamma: The decaying factor associated with the scheduler.
            momentum: The momentum that accelerates the training rate schedule.
            main_log_dir: The directory where logs are stored.
            device: The device to be used for training.
            split: The dataset split to be used. In this example, we have used
                "train".
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
        self.summary = {}
        self.cfg.convert_to_tf_names('pipeline')

    def run_inference(self, data):
        """Run the inference using the data passed."""
        model = self.model
        log.info("running inference")

        model.inference_begin(data)

        while True:
            inputs = model.inference_preprocess()
            results = model(inputs, training=False)
            if model.inference_end(results):
                break

        metric = SemSegMetric()
        metric.update(
            tf.convert_to_tensor(model.inference_result['predict_scores']),
            tf.convert_to_tensor(data['label']))
        log.info(f"Accuracy : {metric.acc()}")
        log.info(f"IoU : {metric.iou()}")

        return model.inference_result

    def run_test(self):
        """Run the test using the data passed."""
        model = self.model
        dataset = self.dataset
        cfg = self.cfg

        self.load_ckpt(model.cfg.ckpt_path)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        record_summary = cfg.get('summary').get('record_for', [])
        log.info("Started testing")

        metric = SemSegMetric()
        Loss = SemSegLoss(self, model, dataset)

        test_split = dataset.get_split('test')
        for idx in tqdm(range(len(test_split)), desc='test'):
            attr = test_split.get_attr(idx)
            data = test_split.get_data(idx)
            results = self.run_inference(data)
            scores, labels = Loss.filter_valid_label(results['predict_scores'],
                                                     data['label'])
            metric.update(scores, labels)
            dataset.save_test_result(results, attr)
            # Save only for the first batch
            if 'test' in record_summary and 'test' not in self.summary:
                self.summary['test'] = self.get_3d_summary(tf.convert_to_tensor(
                    results['predict_scores']),
                                                           data,
                                                           0,
                                                           save_gt=False)

        accs = metric.acc()
        ious = metric.iou()

        log.info("Per class Accuracy : {}".format(accs[:-1]))
        log.info("Per class IOUs : {}".format(ious[:-1]))
        log.info("Overall Accuracy : {:.3f}".format(accs[-1]))
        log.info("Overall IOU : {:.3f}".format(ious[-1]))

    def run_train(self):
        """Run model training."""
        model = self.model
        dataset = self.dataset

        cfg = self.cfg

        log.info(model)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        Loss = SemSegLoss(self, model, dataset)
        self.metric_train = SemSegMetric()
        self.metric_val = SemSegMetric()

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
        record_summary = cfg.get('summary').get('record_for', [])
        self.optimizer = model.get_optimizer(cfg)

        is_resume = model.cfg.get('is_resume', True)
        self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

        for epoch in range(0, cfg.max_epoch + 1):
            log.info("=== EPOCH {}/{} ===".format(epoch, cfg.max_epoch))
            # --------------------- training
            self.metric_train.reset()
            self.metric_val.reset()
            self.losses = []
            step = 0

            for inputs in tqdm(train_loader, total=len_train, desc='training'):
                with tf.GradientTape(persistent=True) as tape:
                    results = model(inputs, training=True)

                    loss, gt_labels, predict_scores = model.get_loss(
                        Loss, results, inputs)

                if len(predict_scores.shape) < 2:
                    continue

                if predict_scores.shape[0] == 0:
                    continue
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

                self.metric_train.update(predict_scores, gt_labels)
                self.losses.append(loss.numpy())

                if 'train' in record_summary and step == 0:
                    self.summary['train'] = self.get_3d_summary(
                        results, inputs, epoch)
                step = step + 1

            # --------------------- validation
            self.valid_losses = []
            step = 0

            for inputs in tqdm(valid_loader, total=len_val, desc='validation'):
                with tf.GradientTape() as tape:
                    results = model(inputs, training=False)
                    loss, gt_labels, predict_scores = model.get_loss(
                        Loss, results, inputs)

                if len(predict_scores.shape) < 2:
                    continue

                self.metric_val.update(predict_scores, gt_labels)
                self.valid_losses.append(loss.numpy())

                if 'valid' in record_summary and step == 0:
                    self.summary['valid'] = self.get_3d_summary(
                        results, inputs, epoch)
                step = step + 1

            self.save_logs(writer, epoch)

            if epoch % cfg.save_ckpt_freq == 0 or epoch == cfg.max_epoch:
                self.save_ckpt(epoch)

    def get_3d_summary(self, results, input_data, epoch, save_gt=True):
        """
        Create visualization for network inputs and outputs.

        Args:
            results: Model output (see below).
            input_data: Model input (see below).
            epoch (int): step
            save_gt (bool): Save ground truth (for 'train' or 'valid' stages).

        RandLaNet:
            results (Tensor(B, N, C)): Prediction scores for all classes.
            input_data (Tuple): Batch of pointclouds and labels.
                input_data[0] (Tensor(B,N,3), float) : points
                input_data[-1] (Tensor(B,N), int) : labels

        SparseConvUNet:
            results (Tensor(SN, C)): Prediction scores for all classes. SN is
                total points in the batch.
            input_data (Dict): Batch of pointclouds and labels. Keys should be:
                'point' [Tensor(SN,3), float]: Concatenated points.
                'batch_lengths' [Tensor(B,), int]: Number of points in each
                    point cloud of the batch.
                'label' [Tensor(SN,) (optional)]: Concatenated labels.

        Returns:
            [Dict] visualizations of inputs and outputs suitable to save as an
                Open3D for TensorBoard summary.
        """
        if not hasattr(self, "_first_step"):
            self._first_step = epoch
        label_to_names = self.dataset.get_label_to_names()
        cfg = self.cfg.get('summary')
        max_pts = cfg.get('max_pts')
        if max_pts is None:
            max_pts = np.iinfo(np.int32).max
        use_reference = cfg.get('use_reference', False)
        max_outputs = cfg.get('max_outputs', 1)
        input_pcd = []
        gt_labels = []
        predict_labels = []

        def to_sum_fmt(tensor, add_dims=(0, 0), dtype=np.int32):
            np_tensor = tensor.numpy().astype(dtype)
            new_shape = (1,) * add_dims[0] + np_tensor.shape + (
                1,) * add_dims[1]
            return np_tensor.reshape(new_shape)

        # Variable size point clouds
        if self.model.cfg['name'] in ('KPFCNN', 'KPConv'):
            org_inputs = self.model.organise_inputs(input_data)
            predict_labels.append(to_sum_fmt(tf.argmax(results, 1), (0, 1)))
            if self._first_step == epoch or not use_reference:
                pointcloud = org_inputs['points'][0]
                input_pcd.append(
                    to_sum_fmt(pointcloud[:, :3], (0, 0), np.float32))
                if save_gt:
                    gtl = org_inputs['point_labels']
                    gt_labels.append(to_sum_fmt(gtl, (0, 1)))

        # Dict input, variable size point clouds
        elif self.model.cfg['name'] in ('SparseConvUnet', 'PointTransformer'):
            if self.model.cfg['name'] == 'SparseConvUnet':
                row_splits = np.hstack(
                    ((0,), np.cumsum(input_data['batch_lengths'])))
            else:
                row_splits = input_data.get('row_splits',
                                            [0, input_data['point'].shape[0]])
            max_outputs = min(max_outputs, len(row_splits) - 1)
            for k in range(max_outputs):
                blen_k = row_splits[k + 1] - row_splits[k]
                pcd_step = int(np.ceil(blen_k / min(max_pts, blen_k)))
                res_pcd = results[row_splits[k]:row_splits[k + 1]:pcd_step, :]
                predict_labels.append(to_sum_fmt(tf.argmax(res_pcd, 1), (0, 1)))
                if self._first_step != epoch and use_reference:
                    continue
                pointcloud = input_data['point'][
                    row_splits[k]:row_splits[k + 1]:pcd_step]
                pointcloud = tf.convert_to_tensor(pointcloud)
                input_pcd.append(
                    to_sum_fmt(pointcloud[:, :3], (0, 0), np.float32))
                if save_gt:
                    gtl = input_data['label'][
                        row_splits[k]:row_splits[k + 1]:pcd_step]
                    gt_labels.append(to_sum_fmt(gtl, (0, 1)))

        # Fixed size point clouds
        # Tuple input, same size point clouds
        elif self.model.cfg['name'] in ('RandLANet', 'PVCNN'):
            if self.model.cfg['name'] == 'RandLANet':
                if save_gt:
                    pointcloud = input_data[0]  # 0 => input to first layer
                else:  # Structured data during inference
                    pointcloud = tf.expand_dims(
                        tf.convert_to_tensor(input_data['point']), 0)
                    results = tf.expand_dims(results, 0)
            elif self.model.cfg['name'] == 'PVCNN':
                pointcloud = input_data['point']
            pcd_step = int(
                np.ceil(pointcloud.shape[1] /
                        min(max_pts, pointcloud.shape[1])))
            predict_labels = to_sum_fmt(
                tf.argmax(results[:max_outputs, ::pcd_step, :], -1), (0, 1))
            if self._first_step == epoch or not use_reference:
                input_pcd = to_sum_fmt(pointcloud[:max_outputs, ::pcd_step, :3],
                                       (0, 0), np.float32)
                if save_gt:
                    if self.model.cfg['name'] == 'RandLANet':
                        gtl = input_data[-1]
                        if 'int' not in repr(gtl.dtype).lower():
                            raise TypeError(
                                "Labels should be Int types. Received "
                                f"{gtl.dtype}")
                    elif self.model.cfg['name'] == 'PVCNN':
                        gtl = input_data['label']
                    gt_labels = to_sum_fmt(gtl[:max_outputs, ::pcd_step],
                                           (0, 1))
        else:
            raise NotImplementedError(
                "Saving 3D summary for the model "
                f"{self.model.cfg['name']} is not implemented.")

        def get_reference_or(data_tensor):
            if self._first_step == epoch or not use_reference:
                return data_tensor
            return self._first_step

        summary_dict = {
            'semantic_segmentation': {
                "vertex_positions": get_reference_or(input_pcd),
                "vertex_gt_labels": get_reference_or(gt_labels),
                "vertex_predict_labels": predict_labels,
                'label_to_names': label_to_names
            }
        }
        return summary_dict

    def save_logs(self, writer, epoch):
        """Save logs from the training and send results to TensorBoard."""
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

            for stage in self.summary:
                for key, summary_dict in self.summary[stage].items():
                    label_to_names = summary_dict.pop('label_to_names', None)
                    summary3d.add_3d('/'.join((stage, key)),
                                     summary_dict,
                                     epoch,
                                     max_outputs=0,
                                     label_to_names=label_to_names,
                                     logdir=self.tensorboard_dir)

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        """Load a checkpoint. You must pass the checkpoint and indicate if you want
        to resume.
        """
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
        """Save a checkpoint at the passed epoch."""
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


PIPELINE._register_module(SemanticSegmentation, "tf")
