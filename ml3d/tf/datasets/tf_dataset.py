from abc import abstractmethod
from tqdm import tqdm

import tensorflow as tf
import numpy as np
from ml3d.torch.utils import dataset_helper


class TF_Dataset():
    def __init__(self,
                 *args,
                 dataset=None,
                 preprocess=None,
                 transform=None,
                 no_progress: bool = False,
                 **kwargs):
        self.dataset = dataset
        self.preprocess = preprocess
        self.transform = transform

    # def generator()
    def read_data(self, key):
            print(key)
            data = self.dataset.get_data(key)
            print(data)
            return data['point'], data['feat'], data['label']

    def get_loader(self):

        tf_dataset = tf.data.Dataset.range(len(self.dataset))    
        tf_dataset = tf_dataset.map(lambda x : tf.numpy_function(func = self.read_data, inp = [x], Tout = [tf.float32, tf.float32, tf.int32]))

        for a in tf_dataset:
            print(a)
            print("\n")
            break
    #     print(tf_dataset)



from ml3d.torch.utils import Config
from ml3d.datasets import Toronto3D

if __name__ == '__main__':
    config = '../../torch/configs/randlanet_toronto3d.py'
    cfg = Config.load_from_file(config)
    dataset = Toronto3D(cfg.dataset)
    
    tf_data = TF_Dataset(dataset = dataset.get_split('training'))
    loader = tf_data.get_loader()
    # print(loader)
    # loader = SimpleDataset(dataset = dataset.get_split('training'))
    # print(loader)
