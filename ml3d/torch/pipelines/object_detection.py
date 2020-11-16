import torch

from .base_pipeline import BasePipeline
from ...utils import PIPELINE

class ObjectDetection(BasePipeline):
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
        return

    def run_test(self):
        """
        Run testing on test sets.
            
        """
        model = self.model
        device = self.device
    
        model.to(device)
        model.device = device
        model.eval()

        checkpoint = torch.load("/home/prantl/obj_det/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth", map_location=device)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

        #load_state_dict(model, state_dict, strict, logger)

        model.load_state_dict(state_dict)#ckpt['model_state_dict'])

        dataset_split = self.dataset.get_split('test')
        #test_split = TorchDataloader(dataset=dataset_split,
        #                             preprocess=model.preprocess,
        #                             transform=model.transform,
        #                             use_cache=False,
        #                             shuffle=False)

        import numpy as np
        ref_in    = np.load("/home/prantl/obj_det/test_data/test/input.npy")
        ref_out_0 = np.load("/home/prantl/obj_det/test_data/test/outs0.npy")
        ref_out_1 = np.load("/home/prantl/obj_det/test_data/test/outs1.npy")
        ref_out_2 = np.load("/home/prantl/obj_det/test_data/test/outs2.npy")
        
        attr = dataset_split.get_attr(0)
        data = dataset_split.get_data(0)

        #np.testing.assert_allclose(data, ref_in)

        d = torch.tensor([ref_in], dtype=torch.float32, device=self.device) #data['point']

        with torch.no_grad():
            results = model(d)
        
        np.testing.assert_allclose(results[0][0].cpu().numpy(), ref_out_0)
        np.testing.assert_allclose(results[1][0].cpu().numpy(), ref_out_1)
        np.testing.assert_allclose(results[2][0].cpu().numpy(), ref_out_2)

        print(model.inference_end(None, results))

        return

    def run_train(self):
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
                np.load("/home/prantl/obj_det/test_data/train/input00.npy"),
                dtype=torch.float32, device=self.device
            )
        ]
        ref_in_1 = [
            torch.tensor(
                np.load("/home/prantl/obj_det/test_data/train/input10.npy"),
                dtype=torch.float32, device=self.device
            )
        ]
        ref_in_2 = [
            torch.tensor(
                np.load("/home/prantl/obj_det/test_data/train/input20.npy"),
                dtype=torch.float32, device=self.device
            )
        ]
        ref_in_3 = [
            torch.tensor(
                np.load("/home/prantl/obj_det/test_data/train/input3%d.npy"%i),
                dtype=torch.float32, device=self.device
            ) for i in range(6)
        ]
        ref_in_4 = [
            torch.tensor(
            np.load("/home/prantl/obj_det/test_data/train/input4%d.npy"%i),
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

        ref_out_0 = np.load("/home/prantl/obj_det/test_data/train/out0.npy")
        ref_out_1 = np.load("/home/prantl/obj_det/test_data/train/out1.npy")
        ref_out_2 = np.load("/home/prantl/obj_det/test_data/train/out2.npy")
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