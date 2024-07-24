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
import torch
from torch.utils.data import DataLoader, Dataset

#%% Using KPConv


def main():
    # Initializing path to the current directory
    example_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Assigning checkpoint and point cloud directory as variables
    ckpt_path = os.path.join(example_dir, "vis_weights_RandLANet.pth")
    pc_path = os.path.join(example_dir, "BLOK_D_1.npy")
    
    print('Running...')
    # Loading the point cloud into numpy array
    point = np.load(pc_path)[:, 0:3]
    color = np.load(pc_path)[:, 3:] * 255
       
    data = {
        'name': 'my_point_cloud',
        'point': point,
        'color': color,
        'feat': None,
        'label': None,
     }
    
    model = ml3d.models.RandLANet(ckpt_path)
    pipeline_r = ml3d.pipelines.SemanticSegmentation(model)
    print(f"The device is currently running on: {pipeline_r.device}")
    pipeline_r.load_ckpt(model.cfg.ckpt_path)
    print('Running Inference')
    
    # Run inference
    results_r = pipeline_r.run_inference(data)
    print('Inference processed successfully...')
    pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
    print('Prediction...')
    # Fill "unlabeled" value because predictions have no 0 values.
    pred_label_r[0] = 0
    print('So far so good')
        
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    main()

