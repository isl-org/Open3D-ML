#MUST BE INCLUDED TO WORK
import sys
sys.path.append(r"/home/jeevin/Open3D-ML_PRISM/") #directory to the root project

from utils import CustomDataLoader 
import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

def main():
    #Initializing directory paths
    home_directory = os.path.expanduser( '~' )
    cfg_directory = os.path.join(home_directory, "Open3D-ML_PRISM/ml3d/configs")
    cfg_path = os.path.join(cfg_directory, "randlanet_parislille3d.yml")
    cfg = _ml3d.utils.Config.load_from_file(cfg_path)
    #cfg.model['in_channels'] = 3 #3 for default :This model cant take colours
    las_path = r"/home/jeevin/Open3D-ML_PRISM/utils/LOT_BUNGALOW.las"

    testing = CustomDataLoader(cfg,las_path) 
    #testing.VisualizingData() #To visualize raw data prior to inference

    # #Running Inference
    # Xsplit = 16
    # Ysplit = 6
    # Zsplit = 2
    # batches = testing.Domain_Split(Xsplit,Ysplit,Zsplit)
    # pipeline = testing.CreatePipeline()
    # Results = testing.CustomInference(pipeline,batches)
    # testing.SavetoPkl(Results,Dict_num=19) #(Optional) Provide a threshold of the maximum number of points
    # #saved per file. Currently set at 1,100,000 points per file or 19 batches per file.
    
    # testing.PklVisualizer() # Use this to load saved data. (Optional) Provide directory to the saved files.
    # # Comment out the lines associated to running inference above when running the visualizer
    
    #results = testing.load_data(ext='pkl')
    
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
            