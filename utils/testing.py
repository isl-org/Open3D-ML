from custom_load import CustomDataLoader
import numpy as np
import os
import glob

#las_path = r"/mnt/c/Users/zulhe/OneDrive/Documents/Laser Scanning/BLOK_D_"
las_path = r"/mnt/c/Users/zulhe/OneDrive/Documents/Laser Scanning/BLOK_D_1.las"
testing = CustomDataLoader(las_path) #Leave it empty after .npy has been loaded into the current directory
testing.VisualizingData() #Pass in data path or leave it empty if you have the .npy in the current directory
Xsplit = 16
Ysplit = 8
Zsplit = 2
batches = testing.Domain_Split(Xsplit,Ysplit,Zsplit,feat=True) #feat is true if color is required to be included
batch = batches[19]
print(f"\nColor: {batch['feat'][:5,:]}")


# directory = os.path.dirname(os.path.realpath(__file__))
# file_path = glob.glob(os.path.join(directory, "*.npy"))
# Data = np.load(file_path[0])
# color = Data[:13,3:]
# print(f"{color}")
