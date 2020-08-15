
# Open3D-ML
An extension of Open3D to address 3D Machine Learning tasks
This repo is a proposal for the directory structure.

The repo can be used together with the precompiled open3d pip package but will also be shipped with the open3d package.
The file ```examples/train_semantic_seg.py``` contains a working example showing how the repo can be used directly and after it has been integrated in the open3d namespace.

TODO List:
- [x] tensorboard
- [x] strucutred config file
- [x] disentangle config
- [x] support yaml
- [x] validation loader
- [x] re-training
- [x] on-the-fly cached preprocessing
- [x] reorganize the dataloading and caching
- [x] dataset class in torch 
- [ ] support KPConv
- [ ] support S3DIS
- [ ] Tensorflow pipeline
- [ ] semantickitti example data for inference
- [ ] fine-tune training
- [ ] rename ml3d.torch.datasets -> ml3d.torch.dataloaders same for tf
- [ ] replace custom compiled ops with functionality in o3d if possible
- [ ] storage solution for network weights and other large binary data (Git LFS, Google, AWS)
- [ ] check code origins for all files 
  - [ ] ml3d/torch/utils/dataset_helper.py is a copy from Kaolin

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

Some important functions of pipeline, model, and dataset classes,
```
pipeline
	__init__(model, dataset, cfg)
	run_train
	run_test
	run_inference
	compute metrics(iou, acc)

model
	__init__(cfg)
	forward
	preprocess         

dataset
	__init__(cfg)
	save_test_result
	get_sampler(split="training/test/validation")
	get_data(file_path)

config
	__init__()
	load_from_file

	train...
	test...
	network...
	general...
```

## Usage example

First build the project
```bash
bash compile_op.sh
pip install -e .
```


Run demo code
```bash
python examples/train_semantic_seg.py
python examples/test_semantic_seg.py
python examples/inference_semantic_seg.py
```
