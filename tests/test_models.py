import pytest
import numpy as np

def test_randlanet_torch():
    import torch
    import open3d.ml.torch as ml3d
    
    net = ml3d.models.RandLANet(num_points=5000, num_classes=10, dim_input=6)
    net.device = 'cpu'

    data = {
        'point': np.array(np.random.random((1000, 3)), dtype=np.float32),
        'feat': np.array(np.random.random((1000, 3)), dtype=np.float32),
        'label': np.array([np.random.randint(10) for i in range(1000)], dtype=np.int32)
        }
    attr = {
        'split': 'train'
    }

    data = net.preprocess(data, attr)
    inputs = net.transform(data, attr)
    inputs = {
        'xyz': [torch.from_numpy(np.array([item])) for item in inputs['xyz']],
        'neigh_idx': [torch.from_numpy(np.array([item])) for item in inputs['neigh_idx']],
        'sub_idx': [torch.from_numpy(np.array([item])) for item in inputs['sub_idx']],
        'interp_idx': [torch.from_numpy(np.array([item])) for item in inputs['interp_idx']],
        'features': torch.from_numpy(np.array([inputs['features']])),
        'labels': torch.from_numpy(np.array([inputs['labels']]))
    }
    out = net(inputs).detach().numpy()
    
    assert out.shape == (1, 5000, 10)


def test_randlanet_tf():
    import tensorflow as tf
    import open3d.ml.tf as ml3d
    from ml3d.datasets.utils import trans_crop_pc

    net = ml3d.models.RandLANet(num_points=5000, num_classes=10, dim_input=6, num_layers=4)

    data = {
        'point': np.array(np.random.random((1000, 3)), dtype=np.float32),
        'feat': np.array(np.random.random((1000, 3)), dtype=np.float32),
        'label': np.array([np.random.randint(10) for i in range(1000)], dtype=np.int32)
        }
    attr = {
        'split': 'train'
    }

    data = net.preprocess(data, attr)
    pc, feat, label, _ = trans_crop_pc(data['point'], data['feat'], data['label'], data['search_tree'], 0, 5000)

    inputs = net.transform(tf.convert_to_tensor(pc), tf.convert_to_tensor(feat), tf.convert_to_tensor(label))
    for i in range(18):  # num_layers * 4 + 2
        inputs[i] = tf.expand_dims(inputs[i], 0)

    out = net(inputs).numpy()

    assert out.shape == (1, 5000, 10)


# test_randlanet_torch()
# test_randlanet_tf()