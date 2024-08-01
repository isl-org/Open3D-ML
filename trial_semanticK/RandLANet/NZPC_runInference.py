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
from torch.utils.data import DataLoader


# Custom Dataset class for your point cloud data
class CustomPointCloudDataset():
    def __init__(self, point, color, label):
        self.point = point
        self.color = color
        self.label = label

    def __len__(self):
        return len(self.point)

    def __getitem__(self, idx):
        return {
            'point': self.point[idx],
            'color': self.color[idx],
            'label': self.label[idx]
        }



def main():
    example_dir = os.path.dirname(os.path.realpath(__file__)) #Initializing the current directory
    code_name = "000" #Name assigned to each batch together with counter in looping
    batch_size = 65536 #Arbitrary number
    i = 0 #Counter
    vis_points = [] # To compile 'vis_d' dictionary at each looping iteration


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

    # Create custom dataset and dataloader
    dataset = CustomPointCloudDataset(point, color, label)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    print('\nRunning...')
    model = ml3d.models.RandLANet(ckpt_path)
    pipeline_r = ml3d.pipelines.SemanticSegmentation(model)
    print(f"The device is currently running on: {pipeline_r.device}")
    pipeline_r.load_ckpt(model.cfg.ckpt_path)
    print('Running Inference')


    for batch in dataloader:
        i += 1
        name = code_name + str(i)
        print(f"\nIteration number: {i}")
        results_r = pipeline_r.run_inference(batch)
        print('Inference processed successfully...')
        pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
        # Fill "unlabeled" value because predictions have no 0 values.
        pred_label_r[0] = 0
        
        vis_d = {
            "name": name,
            "points": batch["point"],
            "labels": batch["label"],
            "pred": pred_label_r,
                }
        vis_points.append(vis_d)    
        
    v.visualize(vis_points)
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    main()
