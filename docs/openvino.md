# OpenVINO backend

Open3D-ML allows to use [Intel
OpenVINO](https://github.com/openvinotoolkit/openvino) as an optional backend for deep learning models inference.

## Install

Install a compatible version of OpenVINO with:

```sh
pip install -r requirements-openvino.txt
```

## Usage

To enable OpenVINO, wrap a model in `ml3d.models.OpenVINOModel` class. In example,

```python
net = ml3d.models.PointPillars(**cfg.model, device='cpu')
net = ml3d.models.OpenVINOModel(net)
```

Then use `net` as usual.

## Supported hardware

OpenVINO supports Intel CPUs, GPUs and VPUs. By default, model is executed on CPU.
To switch between devices, use `net.to` option:

```python
net.to("cpu")     # CPU device (default)
net.to("gpu")     # GPU device
net.to("myriad")  # VPU device
```

## Supported models

* `RandLA-Net` (tf, torch)
* `KPConv` (tf, torch)
* `PointPillars` (torch)
