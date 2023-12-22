import pytest
import os
import open3d as o3d

if 'PATH_TO_OPEN3D_ML' in os.environ.keys():
    base = os.environ['PATH_TO_OPEN3D_ML']
else:
    base = '.'
    # base = '../Open3D-ML'


@pytest.mark.skipif("not o3d._build_config['BUILD_PYTORCH_OPS']")
def test_integration_torch():
    import torch
    import open3d.ml.torch as ml3d
    from open3d.ml.datasets import S3DIS
    from open3d.ml.utils import Config, get_module
    from open3d.ml.torch.models import RandLANet, KPFCNN
    from open3d.ml.torch.pipelines import SemanticSegmentation
    print(dir(ml3d))

    config = base + '/ml3d/configs/randlanet_toronto3d.yml'
    cfg = Config.load_from_file(config)

    model = ml3d.models.RandLANet(**cfg.model)

    print(model)


@pytest.mark.skipif("not o3d._build_config['BUILD_TENSORFLOW_OPS']")
def test_integration_tf():
    import tensorflow as tf
    import open3d.ml.tf as ml3d
    from open3d.ml.datasets import S3DIS
    from open3d.ml.utils import Config, get_module
    from open3d.ml.tf.models import RandLANet, KPFCNN
    from open3d.ml.tf.pipelines import SemanticSegmentation
    print(dir(ml3d))

    config = base + '/ml3d/configs/randlanet_toronto3d.yml'
    cfg = Config.load_from_file(config)

    model = ml3d.models.RandLANet(**cfg.model)

    print(model)


def test_integration_openvino():
    try:
        from openvino.inference_engine import IECore
    except ImportError:
        return

    if o3d._build_config['BUILD_TORCH_OPS']:
        from open3d.ml.torch.models import OpenVINOModel
    if o3d._build_config['BUILD_TENSORFLOW_OPS']:
        from open3d.ml.tf.models import OpenVINOModel
