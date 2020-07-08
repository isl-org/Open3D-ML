# Open3D-ML
An extension of Open3D to address 3D Machine Learning tasks
This repo is a proposal for the directory structure.

The repo can be used together with the precompiled open3d pip package but will also be shipped with the open3d package.
The file ```examples/inference_segmentation.py``` contains a working example showing how the repo can be used directly and after it has been integrated in the open3d namespace.


## Directories

```
├─ examples             # place for example scripts and notebooks
├─ docs                 # rst files that can be integrated in the open3d docs
├─ ml3d                 # package root dir; this will become open3d/_ml3d in the package
     ├─ datasets        # generic dataset code; will beopen3d.datasets
     ├─ tf              # directory for tensorflow specific code. same structure as ml3d/torch
     ├─ torch           # directory for pytorch specific code
          ├─ datasets   # framework specific dataset code, e.g. wrappers that can make use of the generic dataset code.
          │             # This will be open3d.ml.torch.datasets with open3d.ml.torch as ml3d
          ├─ models     # modules with model code and/or packages for more complicated models.
          ├─ pipelines  # pipelines with categories as subdirectories e.g. detection, segmentation
          
```

## Usage example

Most users will either use tf or torch. The recommended import code is 
```python
import open3d as o3d
import open3d.core as o3c
import open3d.ml.torch as ml3d
# or
import open3d.ml.tf as ml3d

# using ml3d
net 		= ml3d.models.RandLANet(params)
cfg      	= ml3d.datasets.ConfigSemanticKITTI
dataset 	= ml3d.datasets.SemanticKITTI(params)
pipeline 	= ml3d.datasets.pipelines.SemanticSegmentation(model, dataset, cfg)
```

When using this repo directly (e.g. for development) the imports are
```python
import open3d as o3d
import open3d.core as o3c
import ml3d.torch as ml3d
# or
import ml3d.tf as ml3d

# using ml3d
net 		= ml3d.models.RandLANet(params)
cfg      	= ml3d.datasets.ConfigSemanticKITTI
dataset 	= ml3d.datasets.SemanticKITTI(params)
pipeline 	= ml3d.datasets.pipelines.SemanticSegmentation(model, dataset, cfg)
# we don't need layers at the level of models and pipelines
```
> Note that in this case ```ml3d``` will not behave exactly like the packaged version because ```ml3d.layers``` and other functionality from the main repo will be missing. 
> This should not pose a problem because
>  1. we could import the missing functionality in ```ml3d/[tf,torch]/__init__.py```
>  2. we don't need to use layers and ops for **using** models and pipelines. (pipelines and models will use absolute imports internally for getting the layers and ops)
