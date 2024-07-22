#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 02:35:59 2024

@author: helmi
"""

import logging
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import numpy as np
import os
import sys
from os.path import exists, join, dirname


#Initializing path to the current directory
example_dir = os.path.dirname(os.path.realpath(__file__))

#Assigning checkpoint and point cloud directory as variables
#ckpt_path = example_dir + "/vis_weights_KPFCNN.pth"
pc_path = example_dir + "/BLOK_D_1.npy"


print('Running...')
#Loading the point cloud into numpy array
point = np.load(pc_path)[:, 0:3]
color = np.load(pc_path)[:,3:] * 255


data = [
    {
        'name': 'my_point_cloud',
        'points': point,
	'colors': color,
            }
]

vis = ml3d.vis.Visualizer()
vis.visualize(data)
