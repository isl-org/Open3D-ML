import tensorflow as tf

from tqdm import tqdm
from ...utils import Cache, get_hash


class TFDataloader():
    """This class allows you to load datasets for a TensorFlow framework.

    **Example:**
        This example loads the SemanticKITTI dataset using the a point
        cloud to the visualizer::

            import tensorflow as tf
            from ..dataloaders import TFDataloader

            train_split = TFDataloader(dataset=tf.dataset.get_split('training'),
                            model=model,
                            use_cache=tf.dataset.cfg.use_cache,
                            steps_per_epoch=tf.dataset.cfg.get(
                            'steps_per_epoch_train', None))
    """

    def __init__(self,
                 *args,
                 dataset=None,
                 model=None,
                 use_cache=True,
                 steps_per_epoch=None,
                 preprocess=None,
                 transform=None,
                 get_batch_gen=None,
                 **kwargs):
        """Initializes the object, and includes the following steps:

         * Checks if preprocess is available. If yes, then uses the preprocessed data.
         * Checks if cache is used. If not, then uses data from the cache.

        Args:
            dataset: The 3DML dataset object. You can use the base dataset,
                sample datasets, or a custom dataset.
            model: 3DML model object.
            use_cache: Indicates if preprocessed data should be cached.
            steps_per_epoch: The number of steps per epoch that indicates the
                batches of samples to train. If it is None, then the step number
                will be the number of samples in the data.
            preprocess: The model's preprocess method.
            transform: The model's transform method.
            get_batch_gen: <NTD>
        """
        self.dataset = dataset
        self.model = model
        if model is not None:
            self.preprocess = model.preprocess
            self.transform = model.transform
            self.get_batch_gen = model.get_batch_gen
            self.model_cfg = model.cfg
        else:
            self.preprocess = preprocess
            self.transform = transform
            self.get_batch_gen = get_batch_gen

        self.steps_per_epoch = steps_per_epoch

        if self.preprocess is not None and use_cache:
            cache_dir = getattr(dataset.cfg, 'cache_dir')

            assert cache_dir is not None, 'cache directory is not given'

            self.cache_convert = Cache(self.preprocess,
                                       cache_dir=cache_dir,
                                       cache_key=get_hash(
                                           repr(self.preprocess)[:-15]))

            uncached = [
                idx for idx in range(len(dataset)) if dataset.get_attr(idx)
                ['name'] not in self.cache_convert.cached_ids
            ]
            if len(uncached) > 0:
                print("cache key : {}".format(repr(self.preprocess)[:-15]))
                for idx in tqdm(range(len(dataset)), desc='preprocess'):
                    attr = dataset.get_attr(idx)
                    data = dataset.get_data(idx)
                    name = attr['name']

                    self.cache_convert(name, data, attr)

        else:
            self.cache_convert = None
        self.split = dataset.split
        self.pc_list = dataset.path_list
        self.num_pc = len(self.pc_list)

    def read_data(self, index):
        """Returns the data at the index.

        This does one of the following:
         - If cache is available, then gets the data from the cache.
         - If preprocess is available, then gets the preprocessed dataset and then the data.
         - If cache or preprocess is not available, then get the data from the dataset.
        """
        attr = self.dataset.get_attr(index)
        if self.cache_convert:
            data = self.cache_convert(attr['name'])
        elif self.preprocess:
            data = self.preprocess(self.dataset.get_data(index), attr)
        else:
            data = self.dataset.get_data(index)

        return data, attr

    def __getitem__(self, index):
        """Returns the item at index position (idx)."""
        dataset = self.dataset
        index = index % len(dataset)

        data, attr = self.read_data(index)

        if self.transform is not None:
            data = self.transform(data, attr)

        data = {'data': data, 'attr': attr}

        return data

    def __len__(self):
        """Returns the number of steps for an epoch."""
        if self.steps_per_epoch is not None:
            steps_per_epoch = self.steps_per_epoch
        else:
            steps_per_epoch = len(self.dataset)
        return steps_per_epoch

    def get_loader(self, batch_size=1, num_threads=3, transform=True):
        """This constructs the tensorflow dataloader.

        Args:

            batch_size: The batch size to be used for data loading.
            num_threads: The number of parallel threads to be used to data loading.

        Returns:
            The tensorflow dataloader and the number of steps in one epoch.
        """
        gen_func, gen_types, gen_shapes = self.get_batch_gen(
            self, self.steps_per_epoch, batch_size)

        loader = tf.data.Dataset.from_generator(
            gen_func, gen_types,
            gen_shapes).prefetch(tf.data.experimental.AUTOTUNE)

        if transform:
            loader = loader.map(
                map_func=self.transform,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if (self.model is None or 'batcher' not in self.model_cfg.keys() or
                self.model_cfg.batcher == 'DefaultBatcher'):
            loader = loader.batch(batch_size)

        length = len(self.dataset) / batch_size + 1 if len(
            self.dataset) % batch_size else len(self.dataset) / batch_size
        length = length if self.steps_per_epoch is None else self.steps_per_epoch

        return loader, int(length)
