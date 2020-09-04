import pytest
import os

if 'PATH_TO_OPEN3D_ML' in os.environ.keys():
    base = os.environ['PATH_TO_OPEN3D_ML']
else:
    base = '.'
    # base = '../Open3D-ML'

def test_integration_torch():
    import torch
    import open3d.ml.torch as ml3d
    print(dir(ml3d))
    
    config = base + '/ml3d/configs/randlanet_toronto3d.yml'
    cfg = ml3d.utils.Config.load_from_file(config)

    model = ml3d.models.RandLANet(**cfg.model)

    print(model)

def test_integration_tf():
    import tensorflow as tf
    import open3d.ml.tf as ml3d
    print(dir(ml3d))

    config = base + '/ml3d/configs/randlanet_toronto3d.yml'
    cfg = ml3d.utils.Config.load_from_file(config)

    model = ml3d.models.RandLANet(**cfg.model)

    print(model)

test_integration_torch()