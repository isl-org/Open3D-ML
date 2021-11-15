import glob
import logging
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 4], "Requires PyTorch >= 1.4"

if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    setup(
        name='ml3d',
        description='An extension of Open3D for 3D machine learning tasks',
        author='yi',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
    )
