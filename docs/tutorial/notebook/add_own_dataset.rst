.. _add_own_dataset:

Add a custom dataset
-------------------------------------------------
In this tutorial, we will learn how to add a custom dataset that you can use to train a model. Before you add a custom dataset, ensure that you are familiar with the existing datasets in the scripts/download_datasets folder.

Before you start adding your own dataset to Open3D, ensure that you have copied the dataset to the folder <NTA>. We would also presume that you have decided on the model that you want to use for this custom dataset.

For this example, we will use an image dataset DogsAndCats.labels and the RandLANet model.

At a high-level, we will:
- Download and convert dataset
- Create the configuration file
- Read the dataset and use it train model for semantic segmentation


Download and convert dataset
``````````````````````````````````````
You must download the dataset to a folder and extract it. In this example, we are assuming you have extracted the dataset to the dataset\custom_dataset folder.

You must next preprocess the dataset to convert the labels to pointcloud.

.. code-block:: bash

    # Convert labels to pointcloud data
    cd dataset/custom_dataset
    python preprocess.py

Create a configuration file
```````````````````````````````````````
Your configuration file should include all information required to train a model using the custom dataset. To do this, your configuration file should include:
- dataset information
- model information
- pipeline information

For this example, we will use the path to you custom dataset (dataset/custom_dataset). Before you train a model, you must decide the model you want to use. For this example, we will use RandLANet model. To use models, you must import the model from open3d.ml.torch.models.

Below is a sample configuration file that we will use for our dataset.

.. code-block:: bash

    dataset:
        name: CUSTOM
        dataset_path: # dataset/custom_dataset
        cache_dir: ./logs/cache
        class_weights: [3370714, 2856755, 4919229, 318158, 375640,
        478001, 974733, 650464, 791496, 88727, 1284130, 229758, 2272837]
        ignored_label_inds: []
        num_points: 40960
        test_area_idx: 3
        test_result_folder: ./test
        use_cache: False
    model:
        name: RandLANet
        batcher: DefaultBatcher
        ckpt_path: # path/to/your/checkpoint
        dim_feature: 8
        dim_input: 6
        dim_output:
        - 16
        - 64
        - 128
        - 256
        - 512
        grid_size: 0.04
        ignored_label_inds: []
        k_n: 16
        num_classes: 13
        num_layers: 5
        num_points: 40960
        sub_sampling_ratio:
        - 4
        - 4
        - 4
        - 4
        - 2
        t_normalize:
            method: linear
            normalize_points: False
            feat_bias: 0
            feat_scale: 1
    pipeline:
        name: SemanticSegmentation
        adam_lr: 0.01
        batch_size: 2
        learning_rate: 0.01
        main_log_dir: ./logs
        max_epoch: 100
        save_ckpt_freq: 20
        scheduler_gamma: 0.95
        test_batch_size: 3
        train_sum_dir: train_log
        val_batch_size: 2

We will save this file as randlanet_custom.yml in the configs folder.


Read a dataset
``````````````````````````````````````
You must read a dataset and get a split before you can train a model using the dataset. We will read the dataset by specifying its path and then get all splits.

.. code-block:: bash

    #import torch
    import open3d.ml.torch as ml3d

    #Read a dataset by specifying the path. We are also providing the cache directory and training split.
    dataset = ml3d.datasets.Custom3DSplit(dataset_path='../datasets/custom_dataset', cache_dir='./logs/cache',training_split=['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'])
    #Split the dataset for 'training'. You can get the other splits by passing 'validation' or 'test'
    train_split = dataset.get_split('training')

    #view the first 1000 frames using the visualizer
    MyVis = ml3d.vis.Visualizer()
    vis.visualize_dataset(dataset, 'training',indices=range(100))

Now that you have visualized the dataset for training, let us train the model.

You can also create a custom dataset code and add it to `ml3d/datasets`. A Dataset class is independent of an ML framework and has to be derived from
`BaseDataset` defined in `ml3d/datasets/base_dataset.py`. You must implement
another class `MyDatasetSplit` which is used to return data and attributes
for files corresponding to a particular split.

.. code-block:: python

    from .base_dataset import BaseDataset

    class MyDataset(BaseDataset):
        def __init__(self, name="MyDataset"):
            super().__init__(name=name)
            # read file lists.

        def get_split(self, split):
            return MyDatasetSplit(self, split=split)

        def is_tested(self, attr):
            # checks whether attr['name'] is already tested.

        def save_test_result(self, results, attr):
            # save results['predict_labels'] to file.


    class MyDatasetSplit():
        def __init__(self, dataset, split='train'):
            self.split = split
            self.path_list = []
            # collect list of files relevant to split.

        def __len__(self):
            return len(self.path_list)

        def get_data(self, idx):
            path = self.path_list[idx]
            points, features, labels = read_pc(path)
            return {'point': points, 'feat': features, 'label': labels}

        def get_attr(self, idx):
            path = self.path_list[idx]
            name = path.split('/')[-1]
            return {'name': name, 'path': path, 'split': self.split}


Train a model
```````````````````````````````````````
Before you train a model, you must decide the model you want to use. For this example, we will use RandLANet model. To use models, you must import the model from open3d.ml.torch.models.

After you load a dataset, you can initialize any model and then train the model. The following example shows how you can train a model:

.. code-block:: bash

    #Import torch and the model to use for training
    import open3d.ml.torch as ml3d
    from open3d.ml.torch.models import RandLANet
    from open3d.ml.torch.pipelines import SemanticSegmentation

    #Read a dataset by specifying the path. We are also providing the cache directory and training split.
    dataset = ml3d.datasets.custom_dataset(dataset_path='../datasets/custom_dataset', cache_dir='./logs/cache',training_split=['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'])

    #Initialize the RandLANet model with three layers.
    model = RandLANet(dim_input=3)
    pipeline = SemanticSegmentation(model=model, dataset=dataset, max_epoch=100)

    #Run the training
    pipeline.run_train()
