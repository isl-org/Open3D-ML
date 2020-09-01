

# Open3D-ML
An extension of Open3D to address 3D Machine Learning tasks
This repo is a proposal for the directory structure.

The repo can be used together with the precompiled open3d pip package but will also be shipped with the open3d package.
The file ```examples/train_semantic_seg.py``` contains a working example showing how the repo can be used directly and after it has been integrated in the open3d namespace.

TODO List:
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


## Build the project

## Prepare Datasets

## Visualizer
### Investigate a dataset
### Visualize a pointcloud with labels


## Training
### Command line
Train a network by specifying the names of dataset, model, and pipeline (SemanticSegmentation by default). Either a path to the dataset or a config file of the dataset is needed in this case,

```shell
# Initialize a dataset using its path
python examples/train.py ${tf/torch} -p ${PIPELINE_NAME} -m ${MODEL_NAME} \
-d ${DATASET_NAME} --dataset_path ${DATASET_PATH} [optional arguments]

# Initialize a dataset using its config file
python examples/train.py ${tf/torch} -p ${PIPELINE_NAME} -m ${MODEL_NAME} \
-d ${DATASET_NAME} --cfg_dataset ${DATASET_CONFIG_FILE}  [optional arguments]
```

Alternatively, you can run the script using a config file, which contains configs for dataset, model, and pipeline.
```shell
python examples/train.py ${tf/torch} -c ${CONFIG_FILE} [optional arguments]
```

Examples,
```shell
# Train RandLANet on SemanticKITTI for segmantic segmentation (by default)
python examples/train.py torch -m RandLANet \
-d SemanticKITTI --dataset_path ../dataset/SemanticKITTI 

# or
python examples/train.py torch -m RandLANet \
-d SemanticKITTI --cfg_dataset ./ml3d/configs/default_cfgs/semantickitti.yml

# Use a config file to train this model with tensorflow
python examples/train.py tf -c ml3d/configs/randlanet_semantickitti.yml
```
Arguments can be
- `-p, --pipeline`: pipeline name, SemanticSegmentation by default
- `-m, --model`: model name (RnadLANet, KPConv)
- `-d, --dataset`: dataset name (SemanticKITTI, Toronto3D, S3DIS, ParisLille3D, Semantic3D)
- `-c, --c`: config file path (example config files are in in `ml3d/configs/`)
- `--cfg_model`: path to the model's config file
- `--cfg_pipeline`: path to the pipeline's config file
- `--cfg_dataset`: path to the dataset's config file
- `--cfg_model`: path to the model's config file
- `--dataset_path`: path to the dataset
- `--device`: `cpu` or `gpu`

You can also arbitrary arguments in the command line, and the arguments will save in a dictionary and merge with dataset/model/pipeline's existing cfg.
For example, `--foo abc` will add `{"foo": "abc"}`to the cfg dict.

## Inference with pretrained weights
### Test on a dataset
### Test on a pointcloud file
### APIs for inference


## Components of Open3D-ML3D
### pipeline
```
pipeline
	__init__(model, dataset, cfg)
	run_train
	run_test
	run_inference
```
### dataloader
```
dataloader
	__init__(cfg)
	save_test_result
	get_sampler(split="training/test/validation")
	get_data(file_path)
```
### model
```
model
	__init__(cfg)
	forward
	preprocess         
```
