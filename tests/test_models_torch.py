"""PyTorch model tests.

Each CPU+accelerator model test builds the same model and batch on each device,
runs one forward and one backward pass per device, and compares outputs and
gradients (see ``torch_backend_parity.assert_cpu_accelerator_parity``). Models
whose ops have no CPU kernel run on the accelerator only (``@skip_no_accelerator``).

Registered PyTorch models in ``ml3d/torch/models`` and how they are covered here:

+------------------+----------+--------------------------------+--------+-----------------------------+
| Model            | Task     | Dedicated test                 | Parity | Basic                       |
+==================+==========+================================+========+=============================+
| RandLANet        | Semseg   | test_randlanet_torch           | Yes    | Yes (CPU path if no GPU)    |
| KPFCNN           | Semseg   | test_kpconv_torch              | Yes    | Yes; OpenVINO optional      |
| PointPillars     | Obj det  | test_pointpillars_torch        | Yes    | Yes; OpenVINO optional      |
| SparseConvUnet   | Semseg   | test_sparseconvunet_torch      | Yes    | Yes (relaxed atol)          |
| PointRCNN (RPN)  | Obj det  | test_pointrcnn_rpn_torch       | No     | Yes (@skip_no_accelerator)  |
| PointRCNN (RCNN) | Obj det  | test_pointrcnn_rcnn_torch      | No     | Yes (@skip_no_accelerator)  |
| PointTransformer | Semseg   | test_pointtransformer_torch    | Yes    | Yes                         |
| PVCNN            | Semseg   | test_pvcnn_torch               | No     | Yes (@skip_no_accelerator)  |
+------------------+----------+--------------------------------+--------+-----------------------------+

**Parity** — ``assert_cpu_accelerator_parity``: same weights, forward + backward on
CPU and CUDA/XPU, compare outputs and gradients.

**Basic** — forward + backward smoke; accelerator-only rows require CUDA or XPU.

OpenVINOModel (wrapper, not in table): no dedicated test; exercised in
``test_kpconv_torch`` / ``test_pointpillars_torch`` when OpenVINO is available
(CPU torch vs OpenVINO, not CPU vs GPU parity).

Other PyTorch tests outside this file: ``test_integration_torch`` only constructs
RandLANet from a config (no forward pass).

Parity tests need ``BUILD_PYTORCH_OPS`` and a CUDA or XPU device for the
accelerator half. CPU CI (``./ci/run_ci.sh cpu``) does not run this file; use
``./ci/run_ci.sh cuda`` or ``xpu`` locally for the full matrix.
"""

import copy
import os
import sys
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest
import torch

from torch_backend_parity import (assert_cpu_accelerator_parity, move_to_device)

if 'PATH_TO_OPEN3D_ML' in os.environ.keys():
    base = os.environ['PATH_TO_OPEN3D_ML']
elif 'OPEN3D_ML_ROOT' in os.environ.keys():
    base = os.environ['OPEN3D_ML_ROOT']
else:
    # tests/ lives at repo root; works regardless of pytest cwd.
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

try:
    from open3d.ml.torch.models import OpenVINOModel
    openvino_available = True
except Exception:
    openvino_available = False

_ACCEL_ONLY_REASON = (
    "Requires a CUDA or XPU device; no CPU kernel for one or more ops used by "
    "this model (see test comment).")

_accel = None
if o3d.core.cuda.is_available():
    _accel = torch.device('cuda')
elif o3d.core.sycl.is_available() and torch.xpu.is_available():
    # o3d.core.sycl.is_available() is True whenever Open3D's SYCL module was
    # built, even on hosts with no Intel GPU (it falls back to a SYCL CPU
    # device). torch.xpu.is_available() additionally confirms a real XPU
    # device is visible to PyTorch, so this only picks 'xpu' on actual GPU
    # hardware.
    _accel = torch.device('xpu')

skip_no_pytorch_ops = pytest.mark.skipif(
    "not o3d._build_config['BUILD_PYTORCH_OPS']")
skip_no_accelerator = pytest.mark.skipif(_accel is None,
                                         reason=_ACCEL_ONLY_REASON)

# Model.get_loss() only reads Loss.weighted_CrossEntropyLoss. The real SemSegLoss
# derives class weights from a Dataset, which these synthetic samples do not have.
semseg_loss = SimpleNamespace(
    weighted_CrossEntropyLoss=torch.nn.CrossEntropyLoss())


def _bev_box3d():
    """BEVBox3D lives in Open3D-ML (ml3d), not in open3d.ml.datasets."""
    ml_root = (os.environ.get('OPEN3D_ML_ROOT') or
               os.environ.get('PATH_TO_OPEN3D_ML') or base)
    if ml_root not in sys.path:
        sys.path.insert(0, ml_root)
    from ml3d.datasets.utils import BEVBox3D
    return BEVBox3D


def _to_tensor(x):
    if torch.is_tensor(x):
        return x
    return torch.from_numpy(x)


def _pointrcnn_ctor_kwargs(mode):
    """Small synthetic PointRCNN config for fast accelerator-only tests."""
    npoints = 1024
    rpn_backbone = dict(
        in_channels=0,
        SA_config=dict(
            npoints=[256, 64, 16, 4],
            radius=[[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]],
            nsample=[[4, 8], [4, 8], [4, 8], [4, 8]],
            mlps=[[[8, 8, 16], [16, 16, 32]], [[32, 32, 64], [32, 48, 64]],
                  [[64, 98, 128], [64, 98, 128]],
                  [[128, 128, 256], [128, 192, 256]]],
        ),
        fp_mlps=[[64, 64], [128, 128], [256, 256], [256, 256]],
    )
    rpn_cfg = dict(backbone=rpn_backbone, cls_in_ch=64, reg_in_ch=64)
    rcnn_cfg = dict(
        SA_config=dict(npoints=[16, 4, -1],
                       radius=[0.2, 0.4, 100],
                       nsample=[8, 8, 8],
                       mlps=[[64, 64, 64], [64, 64, 128], [128, 128, 256]]),
        target_head=dict(num_points=32),
        in_channels=64,
        xyz_up_layer=[64, 64],
    )
    return dict(classes=['Car'],
                npoints=npoints,
                mode=mode,
                augment={},
                rpn=rpn_cfg,
                rcnn=rcnn_cfg)


def _pointrcnn_synthetic_sample(seed=44):
    BEVBox3D = _bev_box3d()
    rng = np.random.RandomState(seed)
    n = 1200
    points = rng.uniform(-5, 5, size=(n, 4)).astype(np.float32)
    points[:, 3] = 1.0
    calib = {'world_cam': np.eye(4, dtype=np.float32)}
    box = BEVBox3D(center=[0, 0, 3],
                   size=[1.6, 1.5, 3.9],
                   yaw=0.0,
                   label_class='Car',
                   confidence=-1.0,
                   world_cam=calib['world_cam'])
    data = {
        'point': points,
        'calib': calib,
        'bounding_boxes': [box],
    }
    attr = {'split': 'train'}
    return data, attr


@skip_no_pytorch_ops
def test_randlanet_torch():
    import open3d.ml.torch as ml3d

    np.random.seed(11)
    torch.manual_seed(11)

    def make_net(device=torch.device('cpu')):
        net = ml3d.models.RandLANet(num_points=5000,
                                    num_classes=10,
                                    in_channels=6,
                                    ignored_label_inds=[])
        net.device = str(device)
        net.eval()
        return net.to(device)

    net = make_net()
    state = copy.deepcopy(net.state_dict())

    data = {
        'point': np.random.random((1000, 3)).astype(np.float32),
        'feat': np.random.random((1000, 3)).astype(np.float32),
        'label': np.random.randint(10, size=(1000,)).astype(np.int32),
    }
    attr = {'split': 'train'}
    transformed = net.transform(net.preprocess(data, attr), attr)
    inputs = {
        'coords': [
            torch.from_numpy(np.array([item])) for item in transformed['coords']
        ],
        'neighbor_indices': [
            torch.from_numpy(np.array([item]))
            for item in transformed['neighbor_indices']
        ],
        'sub_idx': [
            torch.from_numpy(np.array([item]))
            for item in transformed['sub_idx']
        ],
        'interp_idx': [
            torch.from_numpy(np.array([item]))
            for item in transformed['interp_idx']
        ],
        'features': torch.from_numpy(np.array([transformed['features']])),
        'labels': torch.from_numpy(np.array([transformed['labels']])),
    }

    def run(device):
        model = make_net(device)
        model.load_state_dict(state)
        dev_inputs = move_to_device(inputs, device)
        out = model(dev_inputs)
        loss, _, _ = model.get_loss(semseg_loss, out, {'data': dev_inputs},
                                    device)
        loss.backward()
        return out, model

    out_cpu, _ = assert_cpu_accelerator_parity(run)
    assert out_cpu.detach().numpy().shape == (1, 5000, 10)


@skip_no_pytorch_ops
def test_kpconv_torch():
    import open3d.ml.torch as ml3d

    np.random.seed(22)
    torch.manual_seed(22)

    def make_net(device=torch.device('cpu')):
        net = ml3d.models.KPFCNN(lbl_values=[0, 1, 2, 3, 4, 5],
                                 num_classes=5,
                                 ignored_label_inds=[0],
                                 in_features_dim=5)
        net.device = str(device)
        net.eval()
        return net.to(device)

    net = make_net()
    state = copy.deepcopy(net.state_dict())

    data = {
        'point': np.random.random((1000, 3)).astype(np.float32),
        'feat': np.random.random((1000, 3)).astype(np.float32),
        'label': np.random.randint(5, size=(1000,)).astype(np.int32),
    }
    attr = {'split': 'train'}
    sample = {
        'data': net.transform(net.preprocess(data, attr), attr),
        'attr': attr
    }

    def run(device):
        model = make_net(device)
        model.load_state_dict(state)
        batcher = ml3d.dataloaders.ConcatBatcher(str(device))
        batch = batcher.collate_fn([sample])
        out = model(batch['data'])
        loss, _, _ = model.get_loss(semseg_loss, out, batch, device)
        loss.backward()
        return out, model

    # Deformable kernel point offsets amplify per-parameter gradient differences.
    out_cpu, _ = assert_cpu_accelerator_parity(run, grad_atol=2e-3)
    assert out_cpu.detach().numpy().shape[1] == 5

    if openvino_available:
        batcher = ml3d.dataloaders.ConcatBatcher('cpu')
        inputs = batcher.collate_fn([sample])
        ov_net = ml3d.models.OpenVINOModel(net)
        ov_net.to("cpu")
        ov_out = ov_net(inputs['data']).detach().numpy()
        assert ov_out.shape == out_cpu.detach().numpy().shape
        assert np.max(np.abs(ov_out - out_cpu.detach().numpy())) < 1e-7


@skip_no_pytorch_ops
def test_pointpillars_torch():
    import open3d.ml.torch as ml3d
    from open3d.ml.utils import Config

    BEVBox3D = _bev_box3d()

    np.random.seed(33)
    torch.manual_seed(33)

    cfg_path = os.path.join(base, 'ml3d', 'configs', 'pointpillars_kitti.yml')
    model_cfg = dict(Config.load_from_file(cfg_path).model)
    model_cfg['augment'] = {}

    def make_net(device=torch.device('cpu')):
        net = ml3d.models.PointPillars(**model_cfg, device=str(device))
        net.eval()
        return net.to(device)

    net = make_net()
    state = copy.deepcopy(net.state_dict())

    box = BEVBox3D(center=[10, 0, -1],
                   size=[1.6, 3.9, 1.5],
                   yaw=0.0,
                   label_class='Car',
                   confidence=-1.0)
    data = {
        'point': np.random.uniform(0, 20, size=(2000, 4)).astype(np.float32),
        'calib': None,
        'bounding_boxes': [box],
    }
    attr = {'split': 'train'}
    sample = {
        'data': net.transform(net.preprocess(data, attr), attr),
        'attr': attr
    }
    batches = {}

    def run(device):
        model = make_net(device)
        model.load_state_dict(state)
        batcher = ml3d.dataloaders.ConcatBatcher(str(device),
                                                 model='PointPillars')
        batch = batcher.collate_fn([sample])
        batch.to(device)
        batches[device.type] = batch
        out = model(batch)
        loss = sum(model.get_loss(out, batch).values())
        loss.backward()
        return out, model

    out_cpu, net_cpu = assert_cpu_accelerator_parity(run)
    assert isinstance(net_cpu.inference_end(out_cpu, batches['cpu']), list)

    if openvino_available:
        batcher = ml3d.dataloaders.ConcatBatcher('cpu', model='PointPillars')
        batch = batcher.collate_fn([sample])
        ov_net = ml3d.models.OpenVINOModel(net)
        ov_results = ov_net(batch)
        for out, ref in zip(ov_results, out_cpu):
            assert out.shape == ref.shape
            assert torch.max(torch.abs(out - ref)) < 1e-5


@skip_no_pytorch_ops
def test_sparseconvunet_torch():
    import open3d.ml.torch as ml3d

    np.random.seed(88)
    torch.manual_seed(88)

    def make_net(device=torch.device('cpu')):
        net = ml3d.models.SparseConvUnet(device=str(device),
                                         num_classes=20,
                                         in_channels=3,
                                         augment={},
                                         ignored_label_inds=[])
        net.eval()
        return net.to(device)

    net = make_net()
    state = copy.deepcopy(net.state_dict())

    data = {
        'point': np.random.random((2000, 3)).astype(np.float32),
        'feat': np.random.random((2000, 3)).astype(np.float32),
        'label': np.random.randint(20, size=(2000,)).astype(np.int32),
    }
    attr = {'split': 'train'}
    sample = {
        'data': net.transform(net.preprocess(data, attr), attr),
        'attr': attr
    }

    def run(device):
        model = make_net(device)
        model.load_state_dict(state)
        batcher = ml3d.dataloaders.ConcatBatcher(str(device),
                                                 model='SparseConvUnet')
        batch = batcher.collate_fn([sample])
        batch['data'].to(device)
        out = model(batch['data'])
        loss, _, _ = model.get_loss(semseg_loss, out, batch, device)
        loss.backward()
        return out, model

    # Voxel hashing orders sparse conv accumulations differently per backend.
    assert_cpu_accelerator_parity(run, atol=1e-2)


@skip_no_pytorch_ops
@skip_no_accelerator
def test_pointrcnn_rpn_torch():
    # NOTE: furthest_point_sampling (PointRCNN RPN Pointnet++ backbone) has no CPU
    # kernel -- see cpp/open3d/ml/pytorch/pointnet/SamplingOps.cpp. Forward+backward
    # runs on accelerator only; no CPU/XPU parity baseline exists.
    import open3d.ml.torch as ml3d

    np.random.seed(44)
    torch.manual_seed(44)

    net = ml3d.models.PointRCNN(device=str(_accel),
                                **_pointrcnn_ctor_kwargs('RPN'))
    data, attr = _pointrcnn_synthetic_sample()
    net.train()
    data = net.preprocess(data, attr)
    t_data = net.transform(data, attr)
    sample = {'data': t_data, 'attr': attr}
    batcher = ml3d.dataloaders.ConcatBatcher(str(_accel), model='PointRCNN')
    batch = batcher.collate_fn([sample])
    batch.to(_accel)

    out = net(batch)
    loss_dict = net.get_loss(out, batch)
    total = sum(loss_dict.values())
    total.backward()
    assert any(p.grad is not None for p in net.rpn.parameters())


@skip_no_pytorch_ops
@skip_no_accelerator
def test_pointrcnn_rcnn_torch():
    # NOTE: same furthest_point_sampling / roipool3d CPU limitation as RPN test.
    # RCNN.loss may assert on SYCL if cls_label contains -1 (ambiguous ROI); try two
    # GT boxes to reduce ambiguous labels.
    import open3d.ml.torch as ml3d

    BEVBox3D = _bev_box3d()
    np.random.seed(45)
    torch.manual_seed(45)

    net = ml3d.models.PointRCNN(device=str(_accel),
                                **_pointrcnn_ctor_kwargs('RCNN'))
    data, attr = _pointrcnn_synthetic_sample(seed=45)
    calib = data['calib']
    data['bounding_boxes'] = [
        BEVBox3D(center=[0, 0, 3],
                 size=[1.6, 1.5, 3.9],
                 yaw=0.0,
                 label_class='Car',
                 confidence=-1.0,
                 world_cam=calib['world_cam']),
        BEVBox3D(center=[2, 1, 4],
                 size=[1.6, 1.5, 3.9],
                 yaw=0.5,
                 label_class='Car',
                 confidence=-1.0,
                 world_cam=calib['world_cam']),
    ]
    net.train()
    data = net.preprocess(data, attr)
    t_data = net.transform(data, attr)
    sample = {'data': t_data, 'attr': attr}
    batcher = ml3d.dataloaders.ConcatBatcher(str(_accel), model='PointRCNN')
    batch = batcher.collate_fn([sample])
    batch.to(_accel)

    out = net(batch)
    required = {
        'cls_label', 'reg_valid_mask', 'roi_boxes3d', 'gt_of_rois', 'pts_input'
    }
    assert required.issubset(out.keys())

    loss_dict = net.get_loss(out, batch)
    total = sum(loss_dict.values())
    total.backward()
    assert any(p.grad is not None for p in net.parameters())


@skip_no_pytorch_ops
def test_pointtransformer_torch():
    import open3d.ml.torch as ml3d

    np.random.seed(66)
    torch.manual_seed(66)

    def make_net(device=torch.device('cpu')):
        net = ml3d.models.PointTransformer(device=str(device),
                                           num_classes=13,
                                           in_channels=6,
                                           augment={},
                                           ignored_label_inds=[])
        net.eval()
        return net.to(device)

    net = make_net()
    state = copy.deepcopy(net.state_dict())

    n = 8192
    data = {
        'point': np.random.random((n, 3)).astype(np.float32),
        'feat': np.random.random((n, 3)).astype(np.float32),
        'label': np.random.randint(13, size=(n,)).astype(np.int32),
    }
    attr = {'split': 'train'}
    sample = {
        'data': net.transform(net.preprocess(data, attr), attr),
        'attr': attr,
    }

    def run(device):
        model = make_net(device)
        model.load_state_dict(state)
        batcher = ml3d.dataloaders.ConcatBatcher(str(device),
                                                 model='PointTransformer')
        batch = batcher.collate_fn([sample])
        batch['data'].to(device)
        out = model(batch['data'])
        loss, _, _ = model.get_loss(semseg_loss, out, batch, device)
        loss.backward()
        return out, model

    out_cpu, _ = assert_cpu_accelerator_parity(run)
    assert out_cpu.shape[1] == 13


@skip_no_pytorch_ops
@skip_no_accelerator
def test_pvcnn_torch():
    # NOTE: trilinear_devoxelize has no CPU kernel (TrilinearDevoxelizeOps.cpp).
    # Forward+backward on accelerator only; no CPU parity baseline.
    import open3d.ml.torch as ml3d

    np.random.seed(77)
    torch.manual_seed(77)

    net = ml3d.models.PVCNN(device=str(_accel),
                            num_classes=13,
                            num_points=256,
                            extra_feature_channels=6,
                            augment={},
                            ignored_label_inds=[])
    net.to(_accel)
    net.train()

    samples = []
    for seed in (771, 772):
        np.random.seed(seed)
        data = {
            'point': np.random.random((400, 3)).astype(np.float32),
            'feat': np.random.random((400, 3)).astype(np.float32) * 255.0,
            'label': np.random.randint(13, size=(400,)).astype(np.int32),
        }
        attr = {'split': 'train'}
        data = net.preprocess(data, attr)
        data = net.transform(data, attr)
        samples.append(data)

    points = torch.stack([_to_tensor(s['point']) for s in samples]).to(_accel)
    feats = torch.stack([_to_tensor(s['feat']) for s in samples]).to(_accel)
    labels = torch.stack([_to_tensor(s['label']) for s in samples]).to(_accel)
    inputs = {'point': points, 'feat': feats}

    out = net(inputs)
    loss, _, _ = net.get_loss(semseg_loss, out, {'data': {
        'label': labels
    }}, _accel)
    loss.backward()
    assert any(p.grad is not None for p in net.parameters())
