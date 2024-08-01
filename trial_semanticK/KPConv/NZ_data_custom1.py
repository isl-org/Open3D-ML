#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:00:27 2024

@author: helmi
"""

#!/usr/bin/env python
import logging
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import numpy as np
import os
import sys
from os.path import exists, join, dirname

#%% Using KPConv

#Need to find a way to usd .yml

#Initializing path to the current directory
example_dir = os.path.dirname(os.path.realpath(__file__))

#Assigning checkpoint and point cloud directory as variables
ckpt_path = example_dir + "/vis_weights_KPFCNN.pth"
pc_path = example_dir + "/BLOK_D_1.npy"


print('Running...')
#Loading the point cloud into numpy array
point = np.load(pc_path)[:, 0:3]
color = np.load(pc_path)[:,3:] * 255

#%%

data = {
     'name': 'my_point_cloud',
     'point': point,
     'color': color,
     'feat': None,
     'label': None,
         }

model = ml3d.models.KPFCNN(ckpt_path)
pipeline_k = ml3d.pipelines.SemanticSegmentation(model)
pipeline_k.load_ckpt(model.cfg.ckpt_path)
print('Running Inference')

#%%
results_k = pipeline_k.run_inference(data)
print('Inference processed successfully')
pred_label_k = (results_k['predict_labels'] + 1).astype(np.int32)
print('Prediction...')
# Fill "unlabeled" value because predictions have no 0 values.
pred_label_k[0] = 0
print('So far so good')
