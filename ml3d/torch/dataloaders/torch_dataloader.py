from tqdm import tqdm
from torch.utils.data import Dataset

from ...utils import Cache, get_hash


class TorchDataloader(Dataset):
    """This class allows you to load datasets for a PyTorch framework.

    Example:
        This example loads the SemanticKITTI dataset using the Torch dataloader:

            import torch
            from torch.utils.data import Dataset, DataLoader
            train_split = TorchDataloader(dataset=dataset.get_split('training'))
    """

    def __init__(self,
                 dataset=None,
                 preprocess=None,
                 transform=None,
                 sampler=None,
                 use_cache=True,
                 steps_per_epoch=None,
                 **kwargs):
        """Initialize.

        Args:
            dataset: The 3D ML dataset class. You can use the base dataset, sample datasets , or a custom dataset.
            preprocess: The model's pre-process method.
            transform: The model's transform method.
            use_cache: Indicates if preprocessed data should be cached.
            steps_per_epoch: The number of steps per epoch that indicates the batches of samples to train. If it is None, then the step number will be the number of samples in the data.

        Returns:
            class: The corresponding class.
        """
        self.dataset = dataset
        self.preprocess = preprocess
        self.steps_per_epoch = steps_per_epoch

        if preprocess is not None and use_cache:
            cache_dir = getattr(dataset.cfg, 'cache_dir')
            assert cache_dir is not None, 'cache directory is not given'

            self.cache_convert = Cache(preprocess,
                                       cache_dir=cache_dir,
                                       cache_key=get_hash(repr(preprocess)))

            uncached = [
                idx for idx in range(len(dataset)) if dataset.get_attr(idx)
                ['name'] not in self.cache_convert.cached_ids
            ]
            if len(uncached) > 0:
                for idx in tqdm(range(len(dataset)), desc='preprocess'):
                    attr = dataset.get_attr(idx)
                    name = attr['name']
                    if name in self.cache_convert.cached_ids:
                        continue
                    data = dataset.get_data(idx)
                    # cache the data
                    self.cache_convert(name, data, attr)

        else:
            self.cache_convert = None

        self.transform = transform

        if sampler is not None:
            sampler.initialize_with_dataloader(self)

    def __getitem__(self, index):
        """Returns the item at index position (idx)."""
        dataset = self.dataset
        index = index % len(dataset)

        attr = dataset.get_attr(index)
        if self.cache_convert:
            data = self.cache_convert(attr['name'])
        elif self.preprocess:
            data = self.preprocess(dataset.get_data(index), attr)
        else:
            data = dataset.get_data(index)

        if self.transform is not None:
            data = self.transform(data, attr)

        inputs = {'data': data, 'attr': attr}

        return inputs

    def __len__(self):
        """Returns the number of steps for an epoch."""
        if self.steps_per_epoch is not None:
            steps_per_epoch = self.steps_per_epoch
        else:
            steps_per_epoch = len(self.dataset)
        return steps_per_epoch
