#coding: future_fstrings
import numpy as np
import logging
import sys

from datetime import datetime
from tqdm import tqdm

from os.path import exists, join, isfile, dirname, abspath
from ml3d.utils import make_dir, LogRecord
import tensorflow as tf
import yaml

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
        cfg = self.cfg
        model = self.model

        model.to(device)
        model.eval()

        with torch.no_grad():
            inputs = model.preprocess_inference(points, device)
            scores = model(inputs)
            pred = torch.max(scores.squeeze(0), dim=-1).indices
            # pred    = pred.cpu().data.numpy()

        return pred

    def run_test(self, device):
        #self.device = device
        model = self.model
        dataset = self.dataset
        cfg = self.cfg
        model.to(device)
        model.eval()

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.addHandler(logging.FileHandler(log_file_path))

        test_sampler = dataset.get_sampler(cfg.test_batch_size, 'test')
        test_loader = DataLoader(test_sampler, batch_size=cfg.test_batch_size)
        test_probs = [
            np.zeros(shape=[len(l), self.model.cfg.num_classes],
                     dtype=np.float16) for l in dataset.possibility
        ]

        self.load_ckpt(model.cfg.ckpt_path, False)
        log.info("Model Loaded from : {}".format(model.cfg.ckpt_path))

        test_smooth = 0.98
        epoch = 0

        log.info("Started testing")

        with torch.no_grad():
            while True:
                for batch_data in tqdm(test_loader, desc='test', leave=False):
                    # loader: point_clout, label
                    inputs = model.preprocess(batch_data, device)
                    result_torch = model(inputs)
                    result_torch = torch.reshape(result_torch,
                                                 (-1, model.cfg.num_classes))

                    m_softmax = nn.Softmax(dim=-1)
                    result_torch = m_softmax(result_torch)
                    stacked_probs = result_torch.cpu().data.numpy()

                    stacked_probs = np.reshape(stacked_probs, [
                        cfg.test_batch_size, model.cfg.num_points,
                        model.cfg.num_classes
                    ])

                    point_inds = inputs['input_inds']
                    cloud_inds = inputs['cloud_inds']

                    for j in range(np.shape(stacked_probs)[0]):
                        probs = stacked_probs[j, :, :]
                        inds = point_inds[j, :]
                        c_i = cloud_inds[j][0]
                        test_probs[c_i][inds] = test_smooth * test_probs[c_i][
                            inds] + (1 - test_smooth) * probs

                new_min = np.min(dataset.min_possibility)
                log.info(f"Epoch {epoch:3d}, end. "
                         f"Min possibility = {new_min:.1f}")

                if np.min(dataset.min_possibility) > 0.5:  # 0.5
                    log.info(f"\nReproject Vote #"
                             f"{int(np.floor(new_min)):d}")
                    dataset.save_test_result(
                        test_probs, str(dataset.cfg.test_split_number))
                    log.info(f"{str(dataset.cfg.test_split_number)}"
                             f" finished")

                    return

                epoch += 1

    def run_train(self):
        model = self.model
        dataset = self.dataset

        cfg = self.cfg

        # strategy TODO
        # strategy_override = None
        # strategy = strategy_override or distribution_utils.get_distribution_strategy(
        #   distribution_strategy=flags_obj.distribution_strategy,
        #   num_gpus=flags_obj.num_gpus,
        #   tpu_address=flags_obj.tpu)

        # strategy_scope = distribution_utils.get_strategy_scope(strategy)

        # mnist = tfds.builder('mnist', data_dir=flags_obj.data_dir)
        # if flags_obj.download:
        # mnist.download_and_prepare()

        # mnist_train, mnist_test = datasets_override or mnist.as_dataset(
        #   split=['train', 'test'],
        #   decoders={'image': decode_image()},  # pylint: disable=no-value-for-parameter
        #   as_supervised=True)
        # train_input_dataset = mnist_train.cache().repeat().shuffle(
        #   buffer_size=50000).batch(flags_obj.batch_size)
        # eval_input_dataset = mnist_test.cache().repeat().batch(flags_obj.batch_size)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            cfg.adam_lr, decay_steps=100000, decay_rate=cfg.scheduler_gamma)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model = self.model
        # TODO, custom metrics and loss
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'])


        num_train_examples = mnist.info.splits['train'].num_examples
        train_steps = num_train_examples // flags_obj.batch_size
        train_epochs = flags_obj.train_epochs

        ckpt_full_path = os.path.join(flags_obj.model_dir, 'model.ckpt-{epoch:04d}')
        callbacks = [
          tf.keras.callbacks.ModelCheckpoint(
              ckpt_full_path, save_weights_only=True),
          tf.keras.callbacks.TensorBoard(log_dir=flags_obj.model_dir),
        ]

        num_eval_examples = mnist.info.splits['test'].num_examples
        num_eval_steps = num_eval_examples // flags_obj.batch_size

        history = model.fit(
          train_input_dataset,
          epochs=train_epochs,
          steps_per_epoch=train_steps,
          callbacks=callbacks,
          validation_steps=num_eval_steps,
          validation_data=eval_input_dataset,
          validation_freq=flags_obj.epochs_between_evals)

        export_path = os.path.join(flags_obj.model_dir, 'saved_model')
        model.save(export_path, include_optimizer=False)

        eval_output = model.evaluate(
          eval_input_dataset, steps=num_eval_steps, verbose=2)

        stats = common.build_stats(history, eval_output, callbacks)
        return stats


    def get_batcher(self, device):
        batcher_name = getattr(self.model.cfg, 'batcher')
        if batcher_name == 'DefaultBatcher':
            batcher = DefaultBatcher()
        elif batcher_name == 'ConcatBatcher':
            batcher = ConcatBatcher(device)
        else:
            batcher = None
        return batcher

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
        if exists(ckpt_path):
            #path = max(list((cfg.ckpt_path).glob('*.pth')))
            log.info(f'Loading checkpoint {ckpt_path}')
            ckpt = torch.load(ckpt_path)
            first_epoch = ckpt['epoch'] + 1
            self.model.load_state_dict(ckpt['model_state_dict'])
            if is_train:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            first_epoch = 0
            log.info('No checkpoint')

        return first_epoch

    def save_ckpt(self, path_ckpt, epoch):
        make_dir(path_ckpt)
        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict(),
                 scheduler_state_dict=self.scheduler.state_dict()),
            join(path_ckpt, f'ckpt_{epoch:02d}.pth'))
        log.info(f'Epoch {epoch:3d}: save ckpt to {path_ckpt:s}')

    def filter_valid(self, scores, labels, device):
        valid_scores = scores.reshape(-1, self.model.cfg.num_classes)
        valid_labels = labels.reshape(-1).to(device)

        ignored_bool = torch.zeros_like(valid_labels, dtype=torch.bool)
        for ign_label in self.dataset.cfg.ignored_label_inds:
            ignored_bool = torch.logical_or(ignored_bool,
                                            torch.eq(valid_labels, ign_label))

        valid_idx = torch.where(torch.logical_not(ignored_bool))[0].to(device)

        valid_scores = torch.gather(
            valid_scores, 0,
            valid_idx.unsqueeze(-1).expand(-1, self.model.cfg.num_classes))
        valid_labels = torch.gather(valid_labels, 0, valid_idx)

        # Reduce label values in the range of logit shape
        reducing_list = torch.arange(0,
                                     self.model.cfg.num_classes,
                                     dtype=torch.int64)
        inserted_value = torch.zeros([1], dtype=torch.int64)

        for ign_label in self.dataset.cfg.ignored_label_inds:
            reducing_list = torch.cat([
                reducing_list[:ign_label], inserted_value,
                reducing_list[ign_label:]
            ], 0)
        valid_labels = torch.gather(reducing_list.to(device), 0, valid_labels)

        valid_labels = valid_labels.unsqueeze(0)
        valid_scores = valid_scores.unsqueeze(0).transpose(-2, -1)

        return valid_scores, valid_labels
