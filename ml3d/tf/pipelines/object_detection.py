import tensorflow as tf
import logging
from tqdm import tqdm

from datetime import datetime

from os.path import exists, join

from .base_pipeline import BasePipeline
from ..dataloaders import TFDataloader
from ...utils import make_dir, PIPELINE, LogRecord

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

        model.inference_begin(data)

        while True:
            inputs = model.inference_preprocess()
            results = model(inputs['data'], training=False)
            if model.inference_end(inputs, results):
                break

        return model.inference_result

    def run_test(self):
        model = self.model
        dataset = self.dataset
        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        test_split = TFDataloader(
            dataset=dataset.get_split('test'),
            model=model,
            use_cache=False,  # nothing to cache.
        )

        self.load_ckpt(model.cfg.ckpt_path)

        log.info("Started testing")

        results = []
        for idx in tqdm(range(len(test_split)), desc='test'):
            data = test_split.read_data(idx)[0]
            result = self.run_inference(data)
            results.extend(result)

    def run_train(self):
        raise NotImplementedError()

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)

        if hasattr(self, 'optimizer'):
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                            optimizer=self.optimizer,
                                            model=self.model)
        else:
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=self.model)

        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  train_ckpt_dir,
                                                  max_to_keep=100)

        if ckpt_path is not None:
            self.ckpt.restore(ckpt_path).expect_partial()
            log.info("Restored from {}".format(ckpt_path))
        else:
            self.ckpt.restore(self.manager.latest_checkpoint)

            if self.manager.latest_checkpoint:
                log.info("Restored from {}".format(
                    self.manager.latest_checkpoint))
            else:
                log.info("Initializing from scratch.")


    def save_ckpt(self, epoch):
        save_path = self.manager.save()
        log.info("Saved checkpoint at: {}".format(save_path))

PIPELINE._register_module(ObjectDetection, "tf")
