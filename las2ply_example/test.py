import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d

# construct a dataset by specifying dataset_path
dataset = ml3d.datasets.SemanticKITTI(dataset_path='/path/to/SemanticKITTI/')

# get the 'all' split that combines training, validation and test set
all_split = dataset.get_split('all')

# print the attributes of the first datum
print(all_split.get_attr(0))

# print the shape of the first point cloud
print(all_split.get_data(0)['point'].shape)

# show the first 100 frames using the visualizer
vis = ml3d.vis.Visualizer()
vis.visualize_dataset(dataset, 'all', indices=range(100))