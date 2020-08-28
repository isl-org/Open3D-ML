import pytest


def test_integration_torch():
    import torch
    import open3d.ml.torch as ml3d
    print(dir(ml3d))

    model = ml3d.models.RandLANet

    model = ml3d.models.RandLANet(
        d_in=6,
        d_out=[16, 64, 128, 256, 512],
        d_feature=8,
        num_classes=8,
        num_layers=5,
    )
    print(model)

    pipeline = ml3d.pipelines.SemanticSegmentation(
        model,
        dataset,
        batch_size=1,
        learning_rate=1e-2,
    )
    print(pipelines)
