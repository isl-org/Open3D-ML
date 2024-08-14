import open3d as o3d
import laspy
import numpy as np
import os
import open3d.ml.torch as ml3d
import glob
import math
import sys

class CustomDataLoader():
    def __init__(self,las_path = None):
        
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.folder = os.path.basename(self.dir) 
        self.file_path = os.path.join(self.dir, self.folder + "_Data.npy")
        self.las_path = las_path
        
        if self.las_path != None:
            #Checking if the path to directory is correct
            try:
                abs_path = os.path.abspath(self.las_path)
                las = laspy.read(abs_path)
                points = np.vstack((las.x, las.y, las.z)).transpose()
                
                #Checking if the .las file has color data
                try:
                    red = las.red
                    green = las.green
                    blue = las.blue
                    colors = np.vstack((red, green, blue)).transpose() / 65535.0 #Normalize to 1
                    data = np.hstack((points,colors))
                    np.save(self.file_path, data)
                
                except:
                    data = points
                    np.save(self.file_path, data)
                    print("\nRGB data are not found. Data will only consist of points")
            except:
                print("\nError: Path to directory is incorrect")
                sys.exit()
        else:
            pass
    
    def VisualizingData(self,data_path = None):
        vis = ml3d.vis.Visualizer()
                  
        #Checking if the user has passed input for data_path
        if data_path == None:
            data_path = self.file_path
                       
        try:
            points = np.load(data_path)[:,0:3]
            try:
                print(f"\nPreparing data for visualization...")
                color = np.load(data_path)[:, 3:] 
                data = [
                    {
                        'name'  : self.folder + str('_PC'),
                        'points': points,
                        'color' : color,
                    }
                                  
                    ]
                
                vis.visualize(data)
            
            except:
                print(f"\nNo color found in data.\nRandomizing color for visualization...")
                color = np.random.rand(*points.shape).astype(np.float32) 
                data = [
                    {
                        'name'  : self.folder + str('_PC'),
                        'points': points,
                        'color' : color,
                    
                    }
                    
                ]
                                
                vis.visualize(data)
        except:
            print(f"\nError: {data_path} is not a valid numpy file\n")
        
    
    def Domain_Split(self,Xsplit,Ysplit,Zsplit,feat = False, point_path = None,label_path = None,color_path = None):
        
        if point_path == None:
            try:
                point = np.load(self.file_path)[:,0:3]
                print(f"\nUsing points from {self.file_path} directory")
            except:
                print(f"\nError: No data found in the directory. Please provide data for point")
                sys.exit()
        else:
            try:
                point = np.load(point_path)
            except:
                print(f"\nError: Path to 'point' directory is incorrect")
                sys.exit()
                
        
        if label_path == None:
            label = np.zeros(np.shape(point)[0], dtype = np.int32)
        else:
            try:
                label = np.load(label_path)
            except:
                print(f"\nError: Path to 'label' directory is incorrect")
                sys.exit()
                
                
        if(color_path == None and feat == True):  
            try:
                color = np.load(self.file_path)[:,3:]
                print(f"Using colors from {self.file_path} directory\n")
            except:
                print(f"\nError: No data found in the directory. Please provide data for color")
                sys.exit()
        elif(color_path != None and feat == True):
            try:
                color = np.load(color_path)
            except:
                print(f"\nError: Path to 'color' directory is incorrect")
                sys.exit()
           
                 
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
        x_splitsize = int(x_len // Xsplit)
        y_splitsize = int(y_len // Ysplit)
        z_splitsize = int(z_len // Zsplit)
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
        
        #To allow function to take in None for 'color' in feat or with variable being passed in:
        if feat == False:
            
            for Class_limit in Class_limits:
                
                Condition = point < Class_limit
                InLimit = point[np.all(Condition == True, axis=1)]
                point = point[np.any(Condition == False, axis=1)]
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

        else:
            
            for Class_limit in Class_limits:
                
                Condition = point < Class_limit
                InLimit = point[np.all(Condition == True, axis=1)]
                point = point[np.any(Condition == False, axis=1)]
                InLabel = label[np.all(Condition == True, axis=1)]
                label = label[np.any(Condition == False, axis=1)]
                InColor = color[np.all(Condition == True, axis=1)]
                color = color[np.any(Condition == False, axis=1)]
                
                if len(InLimit) < 10:
                    pass
                
                else:
                    name = Code_name + str(counter)
                    data = {
                        'name': name,
                        'limit': Class_limit,
                        'point': InLimit,
                        'label': InLabel,
                        'feat': InColor
                        }
                    
                    print(f"\nPoint cloud - {data['name']} has been successfully loaded")
                    print(f"\nNumber of Point Cloud: {len(InLimit)}")
                    batches.append(data)
                    counter += 1

        return batches
        
    
#file_path = glob.glob(os.path.join(directory, "*.npy"))

        
        
     
       
 