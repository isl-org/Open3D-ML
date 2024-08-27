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


i=0


def main():
    # Initializing path to the current directory
    example_dir = os.path.dirname(os.path.realpath(__file__))
    name = 'NZ Point Cloud'
    batch_size = 65536
    
    
    # Assigning checkpoint and point cloud directory as variables
    ckpt_path = os.path.join(example_dir, "vis_weights_RandLANet.pth")
    pc_path = os.path.join(example_dir, "BLOK_D_1.npy")
    
    #Setting up the visualization
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)
    
    
    # Loading the point cloud into numpy array
    point = np.load(pc_path)[:, 0:3]
    color = np.load(pc_path)[:, 3:] * 255
    label = np.zeros(np.shape(point)[0], dtype = np.int32)
    
    data = {
        'point': point,
        'color': color,
        'label': label
               
        }
        
    print('Running...')   
    model = ml3d.models.RandLANet(ckpt_path)
    pipeline_r = ml3d.pipelines.SemanticSegmentation(model, 
                                                     batch_size= batch_size, 
                                                     device='cuda',
                                                     name = name)
    print(f"The device is currently running on: {pipeline_r.device}") #Checking
    pipeline_r.load_ckpt(model.cfg.ckpt_path)
    
    print('Running Inference')
    results_r = pipeline_r.run_inference(data)
    print('Inference processed successfully...')
   
    
    pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
    # Fill "unlabeled" value because predictions have no 0 values.
    pred_label_r[0] = 0
        
    vis_d = {
        "name": name,
        "points": point,
        "labels": label,
        "pred": pred_label_r,
    }
            
    v.visualize(vis_d)
        
if __name__ == "__main__":
    i+= 1
    print(f"\nIteration number: {i}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    main()
