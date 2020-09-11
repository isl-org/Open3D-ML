


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


## Usage
### Command line
Train a network by specifying the names of dataset, model, and pipeline (SemanticSegmentation by default). In order to construct a dataset instance, either a path to the dataset or a config file of is needed,

```shell
# Initialize a dataset using its path
python scripts/run.py ${tf/torch} -p ${PIPELINE_NAME} -m ${MODEL_NAME} \
-d ${DATASET_NAME} --dataset_path ${DATASET_PATH} [optional arguments]

# Initialize a dataset using its config file
python scripts/run.py ${tf/torch} -p ${PIPELINE_NAME} -m ${MODEL_NAME} \
-d ${DATASET_NAME} --cfg_dataset ${DATASET_CONFIG_FILE}  [optional arguments]
```

Alternatively, you can run the script using one single config file, which contains configs for dataset, model, and pipeline.
```shell
python scripts/run.py ${tf/torch} -c ${CONFIG_FILE} [optional arguments]
```

Examples,
```shell
# Train RandLANet on SemanticKITTI for segmantic segmentation 
python scripts/run.py torch -m RandLANet \
-d SemanticKITTI --cfg_dataset ml3d/configs/default_cfgs/semantickitti.yml \
--dataset_path ../dataset/SemanticKITTI 


# Use a config file to train this model with tensorflow
python scripts/run.py tf -c ml3d/configs/kpconv_semantickitti.yml \
--dataset_path ../--pipeline.batch_size 2
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

### Python API
Users can also use python apis to read data, perform inference or training. Example code can be found in `examples/demo_api.py`.

First, let's see how to read dataset from a dataset,
```python
from ml3d.datasets import SemanticKITTI
from ml3d.torch.pipelines import SemanticSegmentation 
from ml3d.torch.models import RandLANet
from ml3d.utils import Config, get_module

def demo_dataset():
    # read data from datasets
    # construct a dataset by specifying dataset_path
    dataset = SemanticKITTI(dataset_path="../dataset/SemanticKITTI",
                            use_cahe=True)
    print(dataset.label_to_names)

    # print names of all pointcould
    all_split = dataset.get_split('all')
    for i in range(len(all_split)):
        attr = all_split.get_attr(i)
        print(attr['name'])

    print(dataset.cfg.validation_split)
    # change the validation split
    dataset.cfg.validation_split = ['01']
    validation_split = dataset.get_split('val')
    for i in range(len(validation_split)):
        data = validation_split.get_data(i)
        print(data['point'].shape)
```

Then, user can also test their data with pretrained weights,
```python
def demo_inference():
    # Inference and test example

    Pipeline = get_module("pipeline", "SemanticSegmentation", "torch")
    Model = get_module("model", "RandLANet", "torch")
    Dataset = get_module("dataset", "SemanticKITTI")

    # Initialize using default configuration in 
    # "ml3d/configs/default_cfgs/randlanet.yml"
    RandLANet = Model(
        ckpt_path="../dataset/checkpoints/randlanet_semantickitti.pth")
    # Initialize by specifying config file path
    SemanticKITTI = Dataset(cfg="ml3d/configs/default_cfgs/semantickitti.yml",
                            use_cahe=False)
    pipeline = Pipeline(model=RandLANet, 
                        dataset=SemanticKITTI,
                        device="gpu")
    # start inference
    # get data
    train_split = SemanticKITTI.get_split("train")
    data = train_split.get_data(0)
    # restore weights
    pipeline.load_ckpt(RandLANet.cfg.ckpt_path, False)
    # run inference
    results = pipeline.run_inference(data)
    print(results)
    # start testing
    pipeline.run_test()

```



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
