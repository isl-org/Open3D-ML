# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:00:27 2024

@author: helmi
"""

#!/usr/bin/env python
import logging
import open3d.ml.torch as ml3d  # just switch to open3d.ml.tf for tf usage
import open3d.ml as _ml3d
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import math


# Custom Dataset class for your point cloud data
class CustomPointCloudDataset():
    def __init__(self, point, label):#, color):
        self.point = point
        #self.color = color
        self.label = label

    def __len__(self):
        return len(self.point)

    def __getitem__(self, idx):
        return {
            'point': self.point[idx],
            #'color': self.color[idx],
            'label': self.label[idx]
        }


def Domain_Split(Xsplit,Ysplit,Zsplit,point,label,color):
    x_par = Xsplit
    y_par = Ysplit
    z_par = Zsplit
    xmax = max(point[:,0])
    ymax = max(point[:,1])
    zmax = max(point[:,2])
    xmin = min(point[:,0])
    ymin = min(point[:,1])
    zmin = min(point[:,2])

    dom_max = [xmax, ymax, zmax]
    dom_min = [xmin, ymin, zmin]

    x_len = xmax - xmin
    y_len = ymax - ymin
    z_len = zmax - zmin

    #Partitioning global domain into smaller section
    x_splitsize = int(x_len // x_par)
    y_splitsize = int(y_len // y_par)
    z_splitsize = int(z_len // z_par)
    tot_splitsize = [x_splitsize, y_splitsize, z_splitsize]

    #Create a boundary limit for each sectional domain using:
    dom_lim = []
    for i in range(len(tot_splitsize)):
        box = list(range(math.floor(dom_min[i]), 
                                math.floor(dom_max[i]) + tot_splitsize[i], 
                                tot_splitsize[i]))
        
        if box[-1] < dom_max[i]:
            box[-1] = int(math.ceil(dom_max[i] +1))

        dom_lim.append(box[1:])

    Xlim,Ylim = np.meshgrid(dom_lim[0],dom_lim[1])
    Xlim = Xlim.flatten()
    Ylim = Ylim.flatten()
    two_Dlim = np.vstack((Xlim,Ylim)).T

    z_val = []
    for Zlim in dom_lim[-1]:
        z_val.append(Zlim * np.ones(len(two_Dlim),dtype=int))

    z_val = np.hstack((z_val[0],z_val[1])).T
    two_Dlim = np.vstack((two_Dlim,two_Dlim)) 
    Class_limits = np.column_stack((two_Dlim,z_val))

    # Do a for loop which iterates through all of the point cloud (pcs)
    Code_name = "000"
    batches = []
    counter = 0
    clone_point = point
    label = label
    color = color 
    
    for Class_limit in Class_limits:
        
        Condition = clone_point < Class_limit
        InLimit = clone_point[np.all(Condition == True, axis=1)]
        clone_point = clone_point[np.any(Condition == False, axis=1)]
        InLabel = label[np.all(Condition == True, axis=1)]
        label = label[np.any(Condition == False, axis=1)]
        #InColor = color[np.all(Condition == True, axis=1)]
        #color = color[np.any(Condition == False, axis=1)]
        
        if len(InLimit) < 10:
            pass
        
        else:
            name = Code_name + str(counter)
            data = {
                'name': name,
                'limit': Class_limit,
                'point': InLimit,
                'label': InLabel,
                #'feat': InColor
                }
            
            print(f"\nPoint cloud - {data['name']} has been successfully loaded")
            print(f"\nNumber of Point Cloud: {len(InLimit)}")
            batches.append(data)
            counter += 1

    return batches
    

def main():
    #Initializing current directory and home directory
    example_dir = os.path.dirname(os.path.realpath(__file__)) 
    home_directory = os.path.expanduser( '~' )
        
    # Assigning checkpoint, point cloud & config files as variables
    ckpt_path = os.path.join(example_dir, "vis_weights_RandLANet_parislille3d.pth")
    pc_path = os.path.join(example_dir, "BLOK_D_1.npy")
    cfg_directory = os.path.join(home_directory, "Open3D-ML_PRISM/ml3d/configs")
    cfg_path = os.path.join(cfg_directory, "randlanet_parislille3d.yml")
    cfg = _ml3d.utils.Config.load_from_file(cfg_path)
    cfg.model['in_channels'] = 3 #3 for default :This model cant take colours
        
    #Setting up empty array and number of features used
    vis_points = [] # To compile 'vis_d' dictionary at each looping iteration
    
    #Setting up the visualization
    Paris3D_labels = ml3d.datasets.ParisLille3D.get_label_to_names() #Using SemanticKITTI labels
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(Paris3D_labels.keys()):
        lut.add_label(Paris3D_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)

    # Loading the point cloud into numpy array
    point = np.load(pc_path)[:, 0:3].astype(np.float32)
    color = np.load(pc_path)[:, 3:] * 255
    label = np.zeros(np.shape(point)[0], dtype = np.int32) 
    
    # Splitting point cloud into batches based on regional area
    Xsplit = 8  #Domain partitioning along X-axis
    Ysplit = 6  #Domain partitioning along Y-axis
    Zsplit = 2  #Domain partitioning along Z-axis
    batches = Domain_Split(Xsplit,Ysplit,Zsplit,point,label,color)
    
        
    print('\n\nConfiguring model...')   
    model = ml3d.models.RandLANet(**cfg.model)
    print("Model Configured...")
    pipeline_r = ml3d.pipelines.SemanticSegmentation(model=model, device="gpu", **cfg.pipeline)
    print(f"The device is currently running on: {pipeline_r.device}")
    pipeline_r.load_ckpt(ckpt_path=ckpt_path)
    print('Running Inference...')


    for i,batch in enumerate(batches):
        i += 1
        print(f"\nIteration number: {i}")
        results_r = pipeline_r.run_inference(batch)
        print('Inference processed successfully...')
        print(f"\nResults_r: {results_r['predict_labels'][:13]}")
        pred_label_r = (results_r['predict_labels'] +1).astype(np.int32) #Plus one?
        print(f"\nBatch point type : {type(batch['point'])}")
        # Fill "unlabeled" value because predictions have no 0 values.
        #pred_label_r[0] = 0
        

        vis_d = {
            "name": batch['name'],
            "points": batch['point'],
            "labels": batch['label'],
            "pred": pred_label_r,
                }
        
        pred_val = vis_d["pred"][:13]
        print(f"\nPrediction values: {pred_val}")
        vis_points.append(vis_d)    

    v.visualize(vis_points)
        
   
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(asctime)s - %(module)s - %(message)s",
    )

    main()

"""         0: 'unclassified',
            1: 'ground',
            2: 'building',
            3: 'pole-road_sign-traffic_light',
            4: 'bollard-small_pole',
            5: 'trash_can',
            6: 'barrier',
            7: 'pedestrian',
            8: 'car',
            9: 'natural-vegetation'"""