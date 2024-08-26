#MUST BE INCLUDED TO WORK
import sys
sys.path.insert(0,r"/home/helmi/Open3D-ML/") #directory to the root project

from utils import CustomDataLoader # type: ignore
import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

def main():
    #Initializing directory paths
    home_directory = os.path.expanduser( '~' )
    cfg_directory = os.path.join(home_directory, "Open3D-ML/ml3d/configs")
    cfg_path = os.path.join(cfg_directory, "kpconv_parislille3d.yml")
    cfg = _ml3d.utils.Config.load_from_file(cfg_path)
    #cfg.model['in_channels'] = 3 #3 for default :This model cant take colours
    las_path = r"/mnt/c/Users/zulhe/OneDrive/Documents/Laser Scanning/BLOK_D_1.las"

    testing = CustomDataLoader() 
    #testing.VisualizingData() #To visualize raw data prior to inference

    #Running Inference
    Xsplit = 16
    Ysplit = 6
    Zsplit = 2
    batches = testing.Domain_Split(Xsplit,Ysplit,Zsplit)
    pipeline = testing.CustomConfig(cfg)
    Results = testing.CustomInference(pipeline,batches)
    testing.SavetoPkl(Results,Dict_num=19) #(Optional) Provide a threshold of the maximum number of points
    #saved per file. Currently set at 1,100,000 points per file or 19 batches per file.
    
    testing.PklVisualizer(cfg) # Use this to load saved data. (Optional) Provide directory to the saved files.
    # Comment out the lines associated to running inference above when running the visualizer
        
    
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
            