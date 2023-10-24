import pytest
import os
import numpy as np
import open3d as o3d
try:
    import torch
except ImportError:
    torch = None

if 'PATH_TO_OPEN3D_ML' in os.environ.keys():
    base = os.environ['PATH_TO_OPEN3D_ML']
else:
    base = '.'

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Use first GPU and restrict memory growth.
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.set_memory_growth(gpus[0], True)
except RuntimeError as e:
    print(e)
except ImportError:
    tf = None

try:
    from open3d.ml.torch.models import OpenVINOModel
    openvino_available = True
except:
    openvino_available = False


@pytest.mark.skipif("not o3d._build_config['BUILD_PYTORCH_OPS']")
def test_randlanet_torch():
    import open3d.ml.torch as ml3d

    net = ml3d.models.RandLANet(num_points=5000, num_classes=10, in_channels=6)
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
        'coords': [
            torch.from_numpy(np.array([item])) for item in inputs['coords']
        ],
        'neighbor_indices': [
            torch.from_numpy(np.array([item]))
            for item in inputs['neighbor_indices']
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


@pytest.mark.skipif("not o3d._build_config['BUILD_TENSORFLOW_OPS']")
def test_randlanet_tf():
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

    out = net(inputs, training=False).numpy()

    assert out.shape == (1, 5000, 10)

    if openvino_available:
        ov_net = ml3d.models.OpenVINOModel(net)
        ov_out = ov_net(inputs)
        assert ov_out.shape == out.shape
        assert np.max(np.abs(ov_out - out)) < 1e-6


@pytest.mark.skipif("not o3d._build_config['BUILD_PYTORCH_OPS']")
def test_kpconv_torch():
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

    net.eval()
    out = net(inputs['data']).detach().numpy()

    assert out.shape[1] == 5

    if openvino_available:
        ov_net = ml3d.models.OpenVINOModel(net)
        ov_net.to("cpu")
        ov_out = ov_net(inputs['data']).detach().numpy()
        assert ov_out.shape == out.shape
        assert np.max(np.abs(ov_out - out)) < 1e-7


@pytest.mark.skipif("not o3d._build_config['BUILD_TENSORFLOW_OPS']")
def test_kpconv_tf():
    import open3d.ml.tf as ml3d

    np.random.seed(32)

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

    if openvino_available:
        ov_net = ml3d.models.OpenVINOModel(net)
        ov_out = ov_net(inputs)
        assert ov_out.shape == out.shape
        assert np.max(np.abs(ov_out - out)) < 1e-5


@pytest.mark.skipif("not o3d._build_config['BUILD_PYTORCH_OPS']")
def test_pointpillars_torch():
    import open3d.ml.torch as ml3d
    from open3d.ml.utils import Config

    cfg_path = base + '/ml3d/configs/pointpillars_kitti.yml'
    cfg = Config.load_from_file(cfg_path)

    net = ml3d.models.PointPillars(**cfg.model, device='cpu')

    batcher = ml3d.dataloaders.ConcatBatcher('cpu', model='PointPillars')
    data = {
        'point': np.array(np.random.random((10000, 4)), dtype=np.float32),
        'calib': None,
        'bounding_boxes': [],
    }
    data = net.preprocess(data, {'split': 'test'})
    data = net.transform(data, {'split': 'test'})
    data = batcher.collate_fn([{'data': data, 'attr': {'split': 'test'}}])

    net.eval()
    with torch.no_grad():
        results = net(data)
        boxes = net.inference_end(results, data)
        assert type(boxes) == list

    if openvino_available:
        ov_net = ml3d.models.OpenVINOModel(net)
        ov_results = ov_net(data)

        for out, ref in zip(ov_results, results):
            assert out.shape == ref.shape
            assert torch.max(torch.abs(out - ref)) < 1e-5


@pytest.mark.skipif("not o3d._build_config['BUILD_TENSORFLOW_OPS']")
def test_pointpillars_tf():
    import open3d.ml.tf as ml3d
    from open3d.ml.utils import Config

    cfg_path = base + '/ml3d/configs/pointpillars_kitti.yml'
    cfg = Config.load_from_file(cfg_path)

    net = ml3d.models.PointPillars(**cfg.model, device='cpu')

    data = [
        tf.constant(np.random.random((10000, 4)), dtype=tf.float32), None, None,
        [tf.constant(np.stack([np.eye(4), np.eye(4)], axis=0))]
    ]

    results = net(data, training=False)
    boxes = net.inference_end(results, data)

    assert type(boxes) == list
