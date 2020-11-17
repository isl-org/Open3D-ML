import torch
import logging
from tqdm import tqdm

from datetime import datetime

from os.path import exists, join

from ..modules.metrics import kitti_eval
from .base_pipeline import BasePipeline
from ..dataloaders import TorchDataloader
from ..utils import latest_torch_ckpt
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

    def __init__(
            self,
            model,
            dataset=None,
            name='ObjectDetection',
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

        self.load_ckpt(model.cfg.ckpt_path)

        datset_split = self.dataset.get_split('test')

        log.info("Started testing")

        results = []
        with torch.no_grad():
            for idx in tqdm(range(len(test_split)), desc='test'):
                data = datset_split.get_data(idx)
                result = self.run_inference(data)
                results.append(result)
        
        ap_res, ap_dict = kitti_eval(gt, results, ['Car', 'Pedestrian', 'Cyclist'])
        log.info("test acc: {}".format(
            ap_res))

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
        """
        train_ckpt_dir = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(train_ckpt_dir)

        if ckpt_path is None:
            ckpt_path = latest_torch_ckpt(train_ckpt_dir)
            if ckpt_path is not None and is_resume:
                log.info('ckpt_path not given. Restore from the latest ckpt')
            else:
                log.info('Initializing from scratch.')
                return

        if not exists(ckpt_path):
            raise FileNotFoundError(f' ckpt {ckpt_path} not found')

        log.info(f'Loading checkpoint {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt and hasattr(self, 'optimizer'):
            log.info(f'Loading checkpoint optimizer_state_dict')
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and hasattr(self, 'scheduler'):
            log.info(f'Loading checkpoint scheduler_state_dict')
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        """

    def test_test(self):
        """
        Run testing on test sets.
            
        """
        model = self.model
        device = self.device
    
        model.to(device)
        model.device = device
        model.eval()

        checkpoint = torch.load("/home/lprantl/obj_det/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth", map_location=device)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

        #load_state_dict(model, state_dict, strict, logger)

        model.load_state_dict(state_dict)#ckpt['model_state_dict'])

        #dataset_split = self.dataset.get_split('test')
        #test_split = TorchDataloader(dataset=dataset_split,
        #                             preprocess=model.preprocess,
        #                             transform=model.transform,
        #                             use_cache=False,
        #                             shuffle=False)

        import numpy as np
        ref_in    = np.load("/home/lprantl/obj_det/test_data/test/input.npy")
        ref_out_0 = np.load("/home/lprantl/obj_det/test_data/test/outs0.npy")
        ref_out_1 = np.load("/home/lprantl/obj_det/test_data/test/outs1.npy")
        ref_out_2 = np.load("/home/lprantl/obj_det/test_data/test/outs2.npy")
        
        #attr = dataset_split.get_attr(0)
        #data = dataset_split.get_data(0)

        #np.testing.assert_allclose(data, ref_in)

        d = torch.tensor([ref_in], dtype=torch.float32, device=self.device) #data['point']

        with torch.no_grad():
            results = model(d)
        
        np.testing.assert_allclose(results[0][0].cpu().numpy(), ref_out_0)
        np.testing.assert_allclose(results[1][0].cpu().numpy(), ref_out_1)
        np.testing.assert_allclose(results[2][0].cpu().numpy(), ref_out_2)

        print(model.inference_end(None, results))

        return

    def test_train(self):
        """
        Run training on train sets
        """
        model = self.model
        device = self.device
    
        model.to(device)
        model.device = device

        model.train()

        import numpy as np

        ref_in_0 = [
            torch.tensor(
                np.load("/home/lprantl/obj_det/test_data/train/input00.npy"),
                dtype=torch.float32, device=self.device
            )
        ]
        ref_in_1 = [
            torch.tensor(
                np.load("/home/lprantl/obj_det/test_data/train/input10.npy"),
                dtype=torch.float32, device=self.device
            )
        ]
        ref_in_2 = [
            torch.tensor(
                np.load("/home/lprantl/obj_det/test_data/train/input20.npy"),
                dtype=torch.float32, device=self.device
            )
        ]
        ref_in_3 = [
            torch.tensor(
                np.load("/home/lprantl/obj_det/test_data/train/input3%d.npy"%i),
                dtype=torch.float32, device=self.device
            ) for i in range(6)
        ]
        ref_in_4 = [
            torch.tensor(
            np.load("/home/lprantl/obj_det/test_data/train/input4%d.npy"%i),
                dtype=torch.int64, device=self.device
            ) for i in range(6)
        ]

        from ..utils.objdet_helper import LiDARInstance3DBoxes
        for i in range(len(ref_in_3)):
            ref_in_3[i] = LiDARInstance3DBoxes(ref_in_3[i])
        losses = model.get_loss(
            None,
            [ref_in_0, ref_in_1, ref_in_2],
            [ref_in_3, ref_in_4])

        ref_out_0 = np.load("/home/lprantl/obj_det/test_data/train/out0.npy")
        ref_out_1 = np.load("/home/lprantl/obj_det/test_data/train/out1.npy")
        ref_out_2 = np.load("/home/lprantl/obj_det/test_data/train/out2.npy")
        print(ref_out_0)
        print(ref_out_1)
        print(ref_out_2)

        np.testing.assert_allclose(losses['loss_cls'][0].cpu().numpy(), ref_out_0)
        np.testing.assert_allclose(losses['loss_bbox'][0].cpu().numpy(), ref_out_1)
        np.testing.assert_allclose(losses['loss_dir'][0].cpu().numpy(), ref_out_2)

        """for step, inputs in enumerate(tqdm(train_loader, desc='training')):
            results = model(inputs['data'])

            losses = model.get_loss(*results)

            loss =
                losses['losses_cls'] +
                losses['losses_bbox'] +
                losses['losses_dir']

            self.optimizer.zero_grad()
            loss.backward()"""

PIPELINE._register_module(ObjectDetection, "torch")