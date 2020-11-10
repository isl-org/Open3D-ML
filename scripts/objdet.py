
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

import torch

from tqdm import tqdm

Model = _ml3d.utils.get_module("model", "PointPillars", "torch")
Dataset = _ml3d.utils.get_module("dataset", "KITTI")

def main():
    model = Model()

    device = torch.device('cuda:0')
    
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

    dataset = Dataset("/home/prantl/obj_det/mmdetection3d/data/kitti")

    dataset_split = dataset.get_split('test')
    #test_split = TorchDataloader(dataset=dataset_split,
    #                             preprocess=model.preprocess,
    #                             transform=model.transform,
    #                             use_cache=False,
    #                             shuffle=False)

    import numpy as np
    ref_in    = np.load("/home/prantl/obj_det/mmdetection3d/points.npy")
    ref_out_0 = np.load("/home/prantl/obj_det/mmdetection3d/outs0.npy")
    ref_out_1 = np.load("/home/prantl/obj_det/mmdetection3d/outs1.npy")
    ref_out_2 = np.load("/home/prantl/obj_det/mmdetection3d/outs2.npy")
    
    attr = dataset_split.get_attr(0)
    data = dataset_split.get_data(0)

    #np.testing.assert_allclose(data, ref_in)

    d = torch.tensor([ref_in], dtype=torch.float32, device=device) #data['point']

    with torch.no_grad():
        results = model(d)
    
    np.testing.assert_allclose(results[0][0].cpu().numpy(), ref_out_0)
    np.testing.assert_allclose(results[1][0].cpu().numpy(), ref_out_1)
    np.testing.assert_allclose(results[2][0].cpu().numpy(), ref_out_2)

if __name__ == '__main__':
    main()
