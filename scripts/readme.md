# Open3D-ML training

[Open3D-ML](https://github.com/intel-isl/Open3D-ML/) is an extension of Open3D to address 3D Machine Learning tasks.
See the [documentation](https://github.com/intel-isl/Open3D-ML/README.md) for how to:

-  Install Open3D-ML
-  Select a pipeline for your task (or build your own)
-  Download a datasets, or use your own
-  Build and train a model
-  Run inference and visualize results


# Usage
## Command line
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

## Python API
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


