import pytest
import numpy as np


def test_randlanet_torch():
    import torch
    import open3d.ml.torch as ml3d

    net = ml3d.models.RandLANet(num_points=5000, num_classes=10, dim_input=6)
    net.device = 'cpu'

    data = {
        'point':
            np.array(np.random.random((1000, 3)), dtype=np.float32),
        'feat':
            np.array(np.random.random((1000, 3)), dtype=np.float32),
        'label':
            np.array([np.random.randint(10) for i in range(1000)],
                     dtype=np.int32)
    }
    attr = {'split': 'train'}

    data = net.preprocess(data, attr)
    inputs = net.transform(data, attr)
    inputs = {
        'xyz': [torch.from_numpy(np.array([item])) for item in inputs['xyz']],
        'neigh_idx': [
            torch.from_numpy(np.array([item])) for item in inputs['neigh_idx']
        ],
        'sub_idx': [
            torch.from_numpy(np.array([item])) for item in inputs['sub_idx']
        ],
        'interp_idx': [
            torch.from_numpy(np.array([item])) for item in inputs['interp_idx']
        ],
        'features': torch.from_numpy(np.array([inputs['features']])),
        'labels': torch.from_numpy(np.array([inputs['labels']]))
    }
    out = net(inputs).detach().numpy()

    assert out.shape == (1, 5000, 10)


def test_randlanet_tf():
    import tensorflow as tf
    import open3d.ml.tf as ml3d

    net = ml3d.models.RandLANet(num_points=5000,
                                num_classes=10,
                                dim_input=6,
                                num_layers=4)

    data = {
        'point':
            np.array(np.random.random((1000, 3)), dtype=np.float32),
        'feat':
            np.array(np.random.random((1000, 3)), dtype=np.float32),
        'label':
            np.array([np.random.randint(10) for i in range(1000)],
                     dtype=np.int32)
    }
    attr = {'split': 'train'}

    data = net.preprocess(data, attr)
    pc, feat, label, _ = ml3d.datasets.utils.trans_crop_pc(
        data['point'], data['feat'], data['label'], data['search_tree'], 0,
        5000)

    inputs = net.transform(tf.convert_to_tensor(pc), tf.convert_to_tensor(feat),
                           tf.convert_to_tensor(label))
    for i in range(18):  # num_layers * 4 + 2
        inputs[i] = tf.expand_dims(inputs[i], 0)

    out = net(inputs).numpy()

    assert out.shape == (1, 5000, 10)


def test_kpconv_torch():
    import torch
    import open3d.ml.torch as ml3d

    net = ml3d.models.KPFCNN(lbl_values=[0, 1, 2, 3, 4, 5],
                             num_classes=4,
                             ignored_label_inds=[0],
                             in_features_dim=5)
    net.device = 'cpu'

    data = {
        'point':
            np.array(np.random.random((1000, 3)), dtype=np.float32),
        'feat':
            np.array(np.random.random((1000, 3)), dtype=np.float32),
        'label':
            np.array([np.random.randint(5) for i in range(1000)],
                     dtype=np.int32)
    }
    attr = {'split': 'train'}
    batcher = ml3d.dataloaders.ConcatBatcher('cpu')

    data = net.preprocess(data, attr)
    inputs = {'data': net.transform(data, attr), 'attr': attr}
    inputs = batcher.collate_fn([inputs])
    out = net(inputs['data']).detach().numpy()

    assert out.shape[1] == 5


def test_kpconv_tf():
    import tensorflow as tf
    import open3d.ml.tf as ml3d

    net = ml3d.models.KPFCNN(lbl_values=[0, 1, 2, 3, 4, 5],
                             num_classes=4,
                             ignored_label_inds=[0],
                             in_features_dim=5)

    data = {
        'point':
            np.array(np.random.random((10000, 3)), dtype=np.float32),
        'feat':
            np.array(np.random.random((10000, 3)), dtype=np.float32),
        'label':
            np.array([np.random.randint(5) for i in range(10000)],
                     dtype=np.int32)
    }
    attr = {'split': 'train'}

    data = net.preprocess(data, attr)
    p_list = tf.convert_to_tensor(data['point'][:1000])
    c_list = tf.convert_to_tensor(
        np.concatenate([data['point'][:1000], data['feat'][:1000]], axis=1))
    pl_list = tf.convert_to_tensor(data['label'][:1000])

    pi_list = tf.convert_to_tensor(
        np.array([i for i in range(1000)], dtype=np.int32))
    ci_list = tf.convert_to_tensor(np.array([0], dtype=np.int32))

    inputs = net.transform(
        p_list, c_list, pl_list,
        tf.convert_to_tensor(np.array([500, 500], dtype=np.int32)), pi_list,
        ci_list)

    out = net(inputs)

    assert out.shape == (1000, 5)
