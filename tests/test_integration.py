import pytest

def test_integration_torch():
    import torch
    import open3d.ml.torch as ml3d
    print(dir(ml3d))

    config = 'ml3d/torch/configs/randlanet_semantickitti.py'
    cfg         = ml3d.utils.Config.load_from_file(config)

    dataset     = ml3d.datasets.SemanticKITTI(cfg.dataset)

    model       = ml3d.models.RandLANet(cfg.model)

    pipeline    = ml3d.pipelines.SemanticSegmentation(model, dataset, cfg.pipeline)

    device      = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pipeline.run_train(device)
