from custom_load import CustomDataLoader
import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d



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
    home_directory = os.path.expanduser( '~' )
    cfg_directory = os.path.join(home_directory, "Open3D-ML_PRISM/ml3d/configs")
    cfg_path = os.path.join(cfg_directory, "randlanet_parislille3d.yml")
    cfg = _ml3d.utils.Config.load_from_file(cfg_path)
    cfg.model['in_channels'] = 3 #3 for models without colours and 6 for models with colours
    las_path = r"/home/jeevin/Open3D-ML_PRISM/utils/BLOKD1.las"

    testing = CustomDataLoader(las_path=las_path) 
    
    #testing.VisualizingData() #To visualize raw data prior to inference

    #Running Inference

    # Xsplit = 18
    # Ysplit = 6
    # Zsplit = 2
    # batches = testing.Domain_Split(Xsplit,Ysplit,Zsplit)
    # pipeline = testing.CustomConfig(cfg)
    # Results = testing.CustomInference(pipeline,batches)
    # testing.SavetoPkl(Results,Dict_num=19) #(Optional) Provide a threshold of the maximum number of points
    #saved per file. Currently set at 1,100,000 points per file or 19 batches per file.

    
    # testing.PklVisualizer(cfg) # Use this to load saved data. (Optional) Provide directory to the saved files.
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
            