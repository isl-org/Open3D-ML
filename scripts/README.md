# Scripts

## `semseg.py`

This script creates and trains a pipeline (SemanticSegmentation by default).
To define the dataset you can pass the path to the dataset or the path to a
config file as shown below.
```shell
# Initialize a dataset using its path
python scripts/semseg.py {tf|torch} -p PIPELINE_NAME -m MODEL_NAME \
-d DATASET_NAME --dataset_path DATASET_PATH [optional arguments]

# Initialize a dataset using its config file
python scripts/semseg.py {tf|torch} -p PIPELINE_NAME -m MODEL_NAME \
-d DATASET_NAME --cfg_dataset DATASET_CONFIG_FILE  [optional arguments]
```

Alternatively, you can run the script using one single config file, which 
contains configs for dataset, model, and pipeline.
```shell
python scripts/semseg.py {tf|torch} -c CONFIG_FILE [optional arguments]
```

### Examples
```shell
# Train RandLANet on SemanticKITTI for segmantic segmentation 
python scripts/semseg.py torch -m RandLANet \
-d SemanticKITTI --cfg_dataset ml3d/configs/default_cfgs/semantickitti.yml \
--dataset_path ../dataset/SemanticKITTI 


# Use a config file to train this model with tensorflow
python scripts/semseg.py tf -c ml3d/configs/kpconv_semantickitti.yml \
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

You can also add arbitrary arguments in the command line and the arguments will
save in a dictionary and merge with dataset/model/pipeline's existing cfg.
For example, `--foo abc` will add `{"foo": "abc"}`to the cfg dict.

