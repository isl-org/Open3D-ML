.. _train_ss_model_using_tensorflow:

Train a Semantic Segmentation Model Using TensorFlow
-------------------------------------------------
In this tutorial, we will learn how to train a semantic segmentation model using TensorFlow in a Jupyter Notebook. We assume that you are familiar with Jupyter Notebook and have created a folder `notebooks` in a folder that is relative to `ml3d`.

Before you begin, ensure that you have TensorFlow installed. To install a compatible version of TensorFlow, use the requirement file:

.. code-block:: bash

    pip install -r requirements-tensorflow.txt

At a high-level, we will:

- Read a dataset and get a training split. For this example, we will use SemanticKITTI dataset.
- Run a pre-trained model. For this example, we will use the RandLANet model.
- Train a model. We will train a model using the SemanticKITTI dataset and RandLANet model.
- Run an inference and run a test. We will run an inference using the 'training' split that use a pointcloud and display a result. However, a test is run on a pre-defined test set rather than a pass pointcloud.


Read a dataset
``````````````````````````````````````
You can use any dataset available in the ``ml3d.datasets`` dataset namespace. For this example, we will use the SemanticKITTI dataset and visualize it. You can use any of the other dataset to load data. However, you must understand that the parameters may vary for each dataset.

We will read the dataset by specifying its path and then get all splits.

.. code-block:: python

    #Training Semantic Segmentation Model using TensorFlow

    #import tensorflow
    import open3d.ml.tf as ml3d
    
    #Read a dataset by specifying the path. We are also providing the cache directory and training split.
    dataset = ml3d.datasets.SemanticKITTI(dataset_path='../datasets/', cache_dir='./logs/cache',training_split=['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'])
    #Split the dataset for 'training'. You can get the other splits by passing 'validation' or 'test'
    train_split = dataset.get_split('training')
    
    #view the first 1000 frames using the visualizer
    MyVis = ml3d.vis.Visualizer()
    vis.visualize_dataset(dataset, 'training',indices=range(100))

Now that you have visualized the dataset for training, let us train the model.

Train a model
```````````````````````````````````````
TensorFlow maps nearly all of GPU memory by default. This may result in `out_of_memory` error if some of the ops allocate memory independent to tensorflow. You may want to limit memory usage as and when needed by the process. Use following code right after importing tensorflow.

.. code-block:: python

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


Refer to this `link <https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth>`_ for more details.

Before you train a model, you must decide the model you want to use. For this example, we will use RandLANet model. To use models, you must import the model from open3d.ml.tf.models.

After you load a dataset, you can initialize any model and then train the model. The following example shows how you can train a model:

.. code-block:: python

    #Training Semantic Segmentation Model using TensorFlow

    #Import tensorflow and the model to use for training
    import open3d.ml.tf as ml3d
    from open3d.ml.tf.models import RandLANet
    from open3d.ml.tf.pipelines import SemanticSegmentation
    
    #Read a dataset by specifying the path. We are also providing the cache directory and training split.
    dataset = ml3d.datasets.SemanticKITTI(dataset_path='../datasets/', cache_dir='./logs/cache',training_split=['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'])
    #Initialize the RandLANet model with three layers.
    model = RandLANet(dim_input=3)
    pipeline = SemanticSegmentation(model=model, dataset=dataset, max_epoch=100)
    #Run the training
    pipeline.run_train()


Run an inference
```````````````````````````````````````
An inference processes point cloud and displays the results based on the trained model. For this example, we will use a trained RandLANet model. 

This example gets the pipeline, model, and dataset based on our previous training example. It runs the inference based the "train" split and prints the results.

.. code-block:: python

    #Training Semantic Segmentation Model using TensorFlow

    #Import tensorflow and the model to use for training
    import open3d.ml.tf as ml3d
    from open3d.ml.tf.models import RandLANet
    from open3d.ml.tf.pipelines import SemanticSegmentation
    
    #Get pipeline, model, and dataset.
    Pipeline = get_module("pipeline", "SemanticSegmentation", "tf")
    Model = get_module("model", "RandLANet", "tf")
    Dataset = get_module("dataset", "SemanticKITTI")
    
    #Create a checkpoint
    RandLANet = Model(ckpt_path=args.path_ckpt_randlanet)
    SemanticKITTI = Dataset(args.path_semantickitti, use_cache=False)
    pipeline = Pipeline(model=RandLANet, dataset=SemanticKITTI)

    #Get data from the SemanticKITTI dataset using the "train" split
    train_split = SemanticKITTI.get_split("train")
    data = train_split.get_data(0)
    
    #Run the inference
    results = pipeline.run_inference(data)

    #Print the results
    print(results)

Run a test
```````````````````````````````````````
Running a test is very similar to running an inference on Jupyter.

This example gets the pipeline, model, and dataset based on our previous training example. It runs the test based the "train" split.

.. code-block:: python

    #Training Semantic Segmentation Model using TensorFlow

    #Import tensorflow and the model to use for training
    import open3d.ml.tf as ml3d
    from open3d.ml.tf.models import RandLANet
    from open3d.ml.tf.pipelines import SemanticSegmentation
    
    #Get pipeline, model, and dataset.
    Pipeline = get_module("pipeline", "SemanticSegmentation", "tf")
    Model = get_module("model", "RandLANet", "tf")
    Dataset = get_module("dataset", "SemanticKITTI")
    
    #Create a checkpoint
    RandLANet = Model(ckpt_path=args.path_ckpt_randlanet)
    SemanticKITTI = Dataset(args.path_semantickitti, use_cache=False)
    pipeline = Pipeline(model=RandLANet, dataset=SemanticKITTI)

    #Get data from the SemanticKITTI dataset using the "train" split
    train_split = SemanticKITTI.get_split("train")
    data = train_split.get_data(0)
    
    #Run the test
    pipeline.run_test(data)
