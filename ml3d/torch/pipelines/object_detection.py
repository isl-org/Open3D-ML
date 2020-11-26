import torch
import logging
from tqdm import tqdm

from datetime import datetime

from os.path import exists, join

from .base_pipeline import BasePipeline
from ..dataloaders import TorchDataloader
from ..utils import latest_torch_ckpt
from ...utils import make_dir, PIPELINE, LogRecord

from ..modules.metrics import mAP

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
        device = self.device

        model.to(device)
        model.device = device
        model.eval()

        model.inference_begin(data)

        with torch.no_grad():
            while True:
                inputs = model.inference_preprocess()
                results = model(inputs['data'])
                if model.inference_end(inputs, results):
                    break

        return model.inference_result

    def run_test(self):
        model = self.model
        dataset = self.dataset
        device = self.device
        cfg = self.cfg
        model.device = device
        model.to(device)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log.info("DEVICE : {}".format(device))
        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        test_split = TorchDataloader(dataset=dataset.get_split('test'),
                                     preprocess=model.preprocess,
                                     transform=model.transform,
                                     use_cache=dataset.cfg.use_cache,
                                     shuffle=False)

        _data = []
        for i in range(1000):
            pc_path = "/home/prantl/obj_det/mmdetection3d/data/kitti/training/velodyne_reduced/%06d.bin"%(i+1)
            label_path = pc_path.replace('velodyne_reduced',
                                            'label_2').replace('.bin', '.txt')
            calib_path = label_path.replace('label_2', 'calib')

            pc = dataset.read_lidar(pc_path)
            calib = dataset.read_calib(calib_path)
            label = dataset.read_label(label_path, calib)

            data = {
                'point': pc,
                'feat': None,
                'calib': calib,
                'bounding_boxes': label,
            }

            _data.append(model.transform(data, None))

        self.load_ckpt(model.cfg.ckpt_path)

        log.info("Started testing")
        self.test_ious = []

        pred = []
        gt = []
        with torch.no_grad():
            for idx in tqdm(range(len(_data)), desc='test'):
                #data = test_split[idx]
                #if (cfg.get('test_continue', True) and dataset.is_tested(data['attr'])):
                #    continue
                results = self.run_inference(_data[idx])#(data['data'])

                # TODO: replace! temporary solution
                trans = calib['R0_rect'] @ calib['Tr_velo2cam']

                import numpy as np

                def limit_period(val, offset=0.5, period=2*np.pi):
                    return val - np.floor(val / period + offset) * period

                def get_difficulty(height):
                    #TODO
                    if height >= 40:
                        return 0  # Easy
                    elif height >= 25:
                        return 1  # Moderate
                    elif height >= 25:
                        return 2  # Hard
                    else:
                        return -1

                def to_camera(bboxes, trans=None, diff=False):
                    bbox = np.empty((len(bboxes), 7))
                    label = np.empty((len(bboxes),))
                    score = np.empty((len(bboxes),))
                    difficulty = np.empty((len(bboxes)))
                    for i, box in enumerate(bboxes):
                        bbox[i, 0:3] = box.center - [0, 0, box.size[1]/2]
                        bbox[i, 3:6] = box.size
                        bbox[i, 3:6] = bbox[i, 3:6][[2, 1, 0]]
                        bbox[i, 6] = limit_period(np.arcsin(box.front[0])-np.pi)
                        label[i] = box.label_class
                        score[i] = box.confidence
                        if diff:
                            difficulty[i] = box.get_kitti_obj_level()
                        else:
                            difficulty[i] = 0#get_difficulty(box.size[1])
                        if trans is not None:
                            bbox[i, 0:3] = (np.array([*bbox[i, 0:3], 1.0]) @ np.transpose(trans))[:3]

                    result =  {
                        'bbox': bbox,
                        'label': label,
                        'score': score,
                        'difficulty': difficulty
                    }

                    return result

                pred.append(to_camera(results, trans))
                gt.append(to_camera(_data[idx]['bounding_boxes'], trans, True))
                #

        if cfg.get('test_compute_metric', True):
            ap = mAP(pred, gt, [0, 1, 2], [0, 1, 2], [0.25, 0.25, 0.5])
            self.test_mAP.append(ap)

        #dataset.save_test_result(results, attr)

        if cfg.get('test_compute_metric', True):
            log.info("test mAP: {}".format(self.test_mAP))

    def run_train(self):
        raise NotImplementedError()

    def load_ckpt(self, ckpt_path=None, is_resume=True):
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

        self.model.load_state_dict(state_dict)


PIPELINE._register_module(ObjectDetection, "torch")
