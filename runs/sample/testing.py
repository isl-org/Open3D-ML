import sys
sys.path.append(r"/home/jeevin/Open3D-ML_PRISM/") #directory to the root project

from utils import *
import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import pickle
import numpy as np
import json
import pprint

####################################
# BASIC INSTRUCTION:

# CustomDataLoader class was made to serve several purposes:

# 1. CustomDataLoader(las_path) will take in a path which leads to the .las file
#    and it will automatically convert it into a .npy file with a default setting of naming
#    convention of 'the name of current folder'_Data.npy
   
# 2. The VisualizingData function contained in the class will allow a user to visualize the data
#    provided that a path to the .npy file of the point cloud is given. If no input is given,
#    the function will search for a .npy corresponding to the naming convention set in Point 1
#    above in the current directory to be visualized
   
# 3. DomainSplit function takes in parameters such as Xsplit,Ysplit,Zsplit to split the global
#    domain into multiple subdomain by dividing the corresponding axes based on 
#    the specified number given. These number must be in integer format. 'feat' has been 
#    set to False by default if 'color' parameter is not required to be included when running
#    inferences. Path to points and colors are optional to be passed in, however, if nothing is
#    set, the function will search for them in the current directory. Color is excluded if 'feat'
#    is set to False which is its default setting. Change it to True if color is needed when
#    running inference. Path to labels can be given optionally. If nothing is passed for the
#    in label, it will be set as None automatically. This function will return a list of
#    dictionaries.
   
# 4. CustomConfig function takes in a .yml file of the setting of the model. A pretrained
#    model of .pth file is required to be attached in the current directory and the class will
#    automatically search for it to be used in the configuration. A different checkpoint path
#    can also be given as an input into the function if the checkpoint is saved elsewhere.
#    This function will return a configured pipeline.
   
# 5. CustomInference function takes in batches from the DomainSplit and also a pipeline which
#    can be obtained by using the CustomConfig function as stated in Point 4. This function
#    will return the final results of the inference and will open up a visualization tab once
#    inferences are done.

####################################


def main():
    #Initializing directory paths
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sliced_path = os.path.join(dir_path,"Sliced_points.npy")
    file_path = r"/home/jeevin/Open3D-ML_PRISM/results/RandLANet_ParisLille3D_sample/Data.npy"
    
    cfg_directory = os.path.expanduser("~/Open3D-ML_PRISM/ml3d/configs/")
    cfg_file = os.path.join(cfg_directory, "randlanet_parislille3d.yml")
    cfg = _ml3d.utils.Config.load_from_file(cfg_file)
    cfg.model['in_channels'] = 3 #3 for models without colours and 6 for models with colours
    las_path = r"runs/sample/LOT_BUNGALOW.las"
    #las_path = r"/mnt/c/Users/zulhe/OneDrive/Documents/LOT BUNGALOW/LOT_BUNGALOW.las"
    
    # Start = [-5950,-7900.45,50.32]
    # End = [-5950.05,-7870,60.27]
    # tol = 0.05
    
    # Data = np.load(file_path)
    # cross_sec = PostProcess()
    # cross_sec.pc_slice(Start,End,tol,Data,"XZ") #try to change the plane orientation XY,XZ,YZ
    # Visualize = CustomDataLoader(cfg)
    # Visualize.VisualizingData(sliced_path)
       
    testing = CustomDataLoader(cfg=cfg, las_path=las_path)
    # testing.VisualizingData(sliced_path)

   
    #testing.VisualizingData() #To visualize raw data prior to inference

    # Running Inference

    # Xsplit = 6
    # Ysplit = 4
    # Zsplit = 1
    # batches = testing.Domain_Split(Xsplit,Ysplit,Zsplit)
    # pipeline = testing.CreatePipeline()
    # Results = testing.CustomInference(pipeline,batches)
    # testing.SavetoPkl(Results,Dict_num=19) #(Optional) Provide a threshold of the maximum number of points
    # testing.SavetoJson(Results, Dict_num=19)
    # saved per file. Currently set at 1,100,000 points per file or 19 batches per file.

    results = testing.load_data(ext='pkl')

    # results[0]['points'][:2,:]
    testing.SavetoLas(results)
    testing.SavetoPkl(results, Dict_num=19)
    # testing.SavetoJson(results, Dict_num=19)
    testing.PklVisualizer() # Use this to load saved data. (Optional) Provide directory to the saved files.


    #Comment out the lines associated to running inference above when running the visualizer
        
    
if __name__ == "__main__":
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
            