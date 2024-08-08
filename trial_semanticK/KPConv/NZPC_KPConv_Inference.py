
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
import math
import torch


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


def Domain_Split(Xsplit,Ysplit,Zsplit,point,label):
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

    for Class_limit in Class_limits:
        
        Condition = clone_point < Class_limit
        InLimit = clone_point[np.all(Condition == True, axis=1)]
        clone_point = clone_point[np.any(Condition == False, axis=1)]
        InLabel = label[np.all(Condition == True, axis=1)]
        label = label[np.any(Condition == False, axis=1)]
        
        if len(InLimit) < 10:
            pass
        
        else:
            name = Code_name + str(counter)
            data = {
                'name': name,
                'limit': Class_limit,
                'point': InLimit,
                'label': InLabel,
                'feat': None
                }
            
            print(f"\nPoint cloud - {data['name']} has been successfully loaded")
            print(f"\nNumber of Point Cloud: {len(InLimit)}")
            batches.append(data)
            counter += 1

    return batches
    
def custom_collate_fn(batch):
    # Extract 'point' and 'label' from each item in the batch
    points = np.vstack([item['point'] for item in batch])
    labels = np.hstack([item['label'] for item in batch])
    
    # Convert numpy arrays to PyTorch tensors
    points_tensor = torch.tensor(points, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return {'point': points_tensor, 'label': labels_tensor}

def main():
    example_dir = os.path.dirname(os.path.realpath(__file__)) #Initializing the current directory
    vis_points = [] # To compile 'vis_d' dictionary at each looping iteration


    # Assigning checkpoint and point cloud directory as variables
    ckpt_path = example_dir + "/vis_weights_KPFCNN.pth"
    pc_path = example_dir + "/BLOK_D_1.npy"
     
        
    #Setting up the visualization
    kitti_labels = ml3d.datasets.SemanticKITTI.get_label_to_names() #Using SemanticKITTI labels
    v = ml3d.vis.Visualizer()
    lut = ml3d.vis.LabelLUT()
    for val in sorted(kitti_labels.keys()):
        lut.add_label(kitti_labels[val], val)
    v.set_lut("labels", lut)
    v.set_lut("pred", lut)


    # Loading the point cloud into numpy array
    point = np.load(pc_path)[:, 0:3]
    #color = np.load(pc_path)[:, 3:] * 255
    label = np.zeros(np.shape(point)[0], dtype = np.int32)
    
    
    # Splitting point cloud into batches based on regional area
    Xsplit = 16 #Domain partitioning along X-axis
    Ysplit = 8  #Domain partitioning along Y-axis
    Zsplit = 2  #Domain partitioning along Z-axis
    batches = Domain_Split(Xsplit,Ysplit,Zsplit,point,label)
    
    
    print('\n\nConfiguring model...')   
    model = ml3d.models.KPFCNN(ckpt_path=ckpt_path)#,batcher='DefaultBatcher')
    pipeline_k = ml3d.pipelines.SemanticSegmentation(model=model)
    print(f"The device is currently running on: {pipeline_k.device}")
    pipeline_k.load_ckpt(model.cfg.ckpt_path)
    print('Running Inference...')


    for i,batch in enumerate(batches):
        i += 1
        print(f"\nIteration number: {i}")
        
       # Use custom collate function to preprocess the batch
        #processed_batch = custom_collate_fn([batch])
        results_k = pipeline_k.run_inference(batch)
        print('Inference processed successfully...')
        print(f"\nResults_k: {results_k['predict_labels'][:13]}")
        pred_label_k = (results_k['predict_labels'] + 1).astype(np.int32) #Plus one?
        # Fill "unlabeled" value because predictions have no 0 values.
        #pred_label_r[0] = 0
        
        vis_d = {
            "name": batch['name'],
            "points": batch['point'],
            "labels": batch['label'],
            "pred": pred_label_k,
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