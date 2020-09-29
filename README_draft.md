<p align="center">
<img src="https://raw.githubusercontent.com/intel-isl/Open3D/master/docs/_static/open3d_logo_horizontal.png" width="320" />
<span style="font-size: 36px">ML</span>
</p>

# Open3D-ML
[**Getting started**](#getting-started) | [**Repository structure**](#repository-structure) | [**Tasks and Algorithms**](#tasks-and-algorithms) |

An extension of Open3D for 3D machine learning tasks.
This repo builds on top of the Open3D core library and extends it with machine learning tools for 3D data processing.
This repo focuses on applications such as semantic point cloud segmentation and provides pretrained models that can be applied to common tasks as well as pipelines for training.

## Requirements

Open3D-ML is integrated in the Open3D v0.11 python distribution.
To use all of the machine learning functionality you need to have installed PyTorch or TensorFlow.
Open3D v0.11 is compatible with the following versions

 * PyTorch 1.6
 * TensorFlow 2.3
 * CUDA 10.1 (optional)

If you need to use different versions we recommend to [build Open3D from source](http://www.open3d.org/docs/release/compilation.html).
- :warning: TODO add infos of how to compile the ml module to the main repo.


## Getting started

### Installation
We provide pre-built pip packages that include Open3D-ML for Ubuntu 18.04+ with Python 3.6+ that can be installed with
```bash
$ pip install open3d
```

To test the installation use 
```bash
# with PyTorch
$ python -c "import open3d.ml.torch as ml3d"
# or with TensorFlow
$ python -c "import open3d.ml.tf as ml3d"
```

### Reading a dataset

```python
import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d

# construct a dataset by specifying dataset_path
dataset = ml3d.datasets.SemanticKITTI(dataset_path='/path/to/SemanticKITTI/')

# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split('all')

# print the attributes of the first datum
print(all_split.get_attr(0))
# returns  {'name': '00_000000', 
#           'path': '/path/to/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin', 
#           'split': 'all'}

# print the shape of the first point cloud
print(all_split.get_data(0)['point'].shape)
# returns (124668, 3)
```
TODO visualization

### Running a pretrained model
```python
TODO
```

### Training a model
```python
TODO
```


## Repository structure
The core part of Open3D-ML lives in the `ml3d` subfolder, which is integrated into Open3D in the `ml` namespace.
In addition to the core part, the directories `examples` and `scripts` provide supporting scripts for getting started setting with seetting up a training pipeline or running a network on a dataset.

```
├─ docs                   # Markdown and rst files for documentation
├─ examples               # Place for example scripts and notebooks
├─ ml3d                   # Package root dir that is integrated in open3d
     ├─ configs           # Model configuration files
     ├─ datasets          # Generic dataset code; will be integratede as open3d.ml.{tf,torch}.datasets
     ├─ utils             # Framework independent utilities; available as open3d.ml.{tf,torch}.utils
     ├─ vis               # ML specific visualization functions
     ├─ tf                # Directory for TensorFlow specific code. same structure as ml3d/torch.
     │                    # This will be available as open3d.ml.tf 
     ├─ torch             # Directory for PyTorch specific code; available as open3d.ml.torch
          ├─ dataloaders  # Framework specific dataset code, e.g. wrappers that can make use of the 
          │               # generic dataset code.
          ├─ models       # Code for models
          ├─ modules      # Smaller modules, e.g., metrics and losses
          ├─ pipelines    # Pipelines for tasks like semantic segmentation
├─ scripts                # Demo scripts for training and dataset download scripts
```


## Tasks and Algorithms

### Segmentation

The table shows the available models and datasets for the segmentation task


| Model / Dataset | SemanticKITTI | Toronto 3D | Semantic 3D | Paris Lille 3D | S3DIS |
|-----------------|---------------|------------|-------------|----------------|-------|
| RandLA-Net      |       X       |            |             |                |       |
| KPConv          |               |     X      |             |                |       |

