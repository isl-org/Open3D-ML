import copy
import os

import numpy as np
import open3d as o3d
import pytest

try:
    import torch
except ImportError:
    torch = None

from torch_backend_parity import (assert_cpu_accelerator_parity, clone_module_to_device,
                                  move_to_device, parity_accelerator)

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
except Exception:
    openvino_available = False


# Looser tolerance on XPU; CUDA and CPU-only use defaults in helper calls.
def _parity_tolerance():
    accel = parity_accelerator()
    if accel is not None and accel.type == "xpu":
        return 1e-3, 1e-4
    return 1e-4, 1e-5


@pytest.mark.skipif("not o3d._build_config['BUILD_PYTORCH_OPS']")
def test_randlanet_torch():
    import open3d.ml.torch as ml3d

    np.random.seed(11)
    torch.manual_seed(11)

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

    net.eval()
    with torch.no_grad():
        out_cpu = net(inputs)

    assert out_cpu.detach().numpy().shape == (1, 5000, 10)

    state = copy.deepcopy(net.state_dict())

    def forward_on_device(device):
        model = ml3d.models.RandLANet(num_points=5000,
                                      num_classes=10,
                                      in_channels=6)
        model.load_state_dict(state)
        model.device = str(device)
        model.to(device)
        model.eval()
        with torch.no_grad():
            return model(move_to_device(inputs, device))

    rtol, atol = _parity_tolerance()
    assert_cpu_accelerator_parity(
        out_cpu,
        forward_on_device,
        rtol=rtol,
        atol=atol,
    )


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

    np.random.seed(22)
    torch.manual_seed(22)

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

    data = net.preprocess(data, attr)
    transform_data = net.transform(data, attr)
    sample = {'data': transform_data, 'attr': attr}

    def run_forward(device):
        batcher = ml3d.dataloaders.ConcatBatcher(str(device))
        model = clone_module_to_device(net, device)
        batch = batcher.collate_fn([sample])
        with torch.no_grad():
            return model(batch['data'])

    out_cpu = run_forward('cpu')
    assert out_cpu.detach().numpy().shape[1] == 5

    rtol, atol = _parity_tolerance()
    assert_cpu_accelerator_parity(
        out_cpu,
        lambda dev: run_forward(dev),
        rtol=rtol,
        atol=atol,
    )

    if openvino_available:
        batcher = ml3d.dataloaders.ConcatBatcher('cpu')
        inputs = batcher.collate_fn([sample])
        ov_net = ml3d.models.OpenVINOModel(net)
        ov_net.to("cpu")
        ov_out = ov_net(inputs['data']).detach().numpy()
        assert ov_out.shape == out_cpu.detach().numpy().shape
        assert np.max(np.abs(ov_out - out_cpu.detach().numpy())) < 1e-7


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

    np.random.seed(33)
    torch.manual_seed(33)

    cfg_path = base + '/ml3d/configs/pointpillars_kitti.yml'
    cfg = Config.load_from_file(cfg_path)

    net = ml3d.models.PointPillars(**cfg.model, device='cpu')

    data = {
        'point': np.array(np.random.random((10000, 4)), dtype=np.float32),
        'calib': None,
        'bounding_boxes': [],
    }
    data = net.preprocess(data, {'split': 'test'})
    data = net.transform(data, {'split': 'test'})
    sample = {'data': data, 'attr': {'split': 'test'}}

    def run_forward(device):
        dev = device if isinstance(device, torch.device) else torch.device(
            device)
        batcher = ml3d.dataloaders.ConcatBatcher(str(dev),
                                                 model='PointPillars')
        model = clone_module_to_device(net, dev)
        batch = batcher.collate_fn([sample])
        if dev.type != 'cpu':
            batch.to(dev)
        with torch.no_grad():
            return model(batch), batch

    results_cpu, batch_cpu = run_forward('cpu')
    boxes = net.inference_end(results_cpu, batch_cpu)
    assert type(boxes) == list

    rtol, atol = _parity_tolerance()
    assert_cpu_accelerator_parity(
        results_cpu,
        lambda dev: run_forward(dev)[0],
        rtol=rtol,
        atol=atol,
    )

    if openvino_available:
        batcher = ml3d.dataloaders.ConcatBatcher('cpu', model='PointPillars')
        batch = batcher.collate_fn([sample])
        ov_net = ml3d.models.OpenVINOModel(net)
        ov_results = ov_net(batch)
        for out, ref in zip(ov_results, results_cpu):
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
