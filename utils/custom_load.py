import open3d.ml.torch as ml3d #Must be first to import to fix bugs when running OpenGL
import laspy
from pathlib import Path
import numpy as np
import os
import glob
import math
import json
import pickle
import shutil
import sys
import pandas as pd



class CustomDataLoader():
    def __init__(self,cfg,las_path = None):

        self.cfg = cfg
        ## paths ##
        main_path = os.path.abspath(sys.argv[0])   
        #file path where the module is called     
        self.dir = Path(main_path).parent  
        print(f"self.dir : {self.dir}") 

        #results path
        results_dir_str = self.cfg.model['name'] + "_" + self.cfg.dataset['name']+ "_" + str(Path(self.dir).stem)  # results path based on the name of the config
        self.folder = Path.cwd() / "results" / results_dir_str    
        print(f"self.folder : {self.folder}")    
        self.file_path_npy = self.folder / "Data.npy"
        

        #print(f"Path(.) : {str(sorted(Path(main_path).parent.glob('*.pth'))[-1])}")
        if self.folder.exists():
            print(f"Folder already exists, will overwrite files inside")
        else:
            self.folder.mkdir(parents=True, exist_ok=True)
            print(f"Folder created at : {str(self.folder)}")
                
        if las_path is not None:
            #Checking if the path to directory is correct
            self.las_path = Path(las_path).resolve()
            las = laspy.read(self.las_path)
            points = np.vstack((las.x, las.y, las.z)).transpose()
            
            #Checking if the .las file has color data
            try:
                red = las.red
                green = las.green
                blue = las.blue
                colors = np.vstack((red, green, blue)).transpose()/65536 #Normalize to 1
                data = np.hstack((points,colors))
                np.save(self.file_path_npy, data)
            
            except:
                data = points
                np.save(self.file_path_npy, data)
                print("\nRGB data are not found. Data will only consist of points")
            
        else:
            pass
            
    
    def VisualizingData(self,data_path = None):
        
        vis = ml3d.vis.Visualizer()
                                                
        #Checking if the user has passed input for data_path
        if data_path == None:
            data_path = self.file_path_npy
                                         
        
        points = np.load(data_path)[:,0:3]
        print(f"\nPreparing data for visualization...")
        try:
                 
            color = np.load(data_path)[:, 3:] 
            data = [
                    {
                    'name'  : self.folder + str('_PC'),
                    'points': points.astype(np.float32),
                    'color' : color.astype(np.int32),
                    }  
                ]
                                    
            vis.visualize(data)
            
        except:
            print(f"\nNo color found in data.\nRandomizing color for visualization...")
            color = np.random.rand(*points.shape) 
            data = [
                    {
                        'name'  : str(self.folder.stem) + str('_PC'),
                        'points': points.astype(np.float32),
                        'color' : color.astype(np.float32),
                    }
                ]
                         
            vis.visualize(data)
                 
    
    def Domain_Split(self,Xsplit=int,Ysplit=int,Zsplit=int,feat = False, point_path = None,label_path = None,color_path = None):
        
        """
        Domain_Split is a function which takes in parameters such as Xsplit,Ysplit,Zsplit to split the global
        domain into multiple subdomain by dividing the corresponding axes based on 
        the specified number given. These number must be in integer format. 'feat' has been 
        set to False by default if 'color' parameter is not required to be included when running
        inferences. Path to points and colors are optional to be passed in, however, if nothing is
        set, the function will search for them in the current directory. Color is excluded if 'feat'
        is set to False which is its default setting. Change it to True if color is needed when
        running inferences.

        Parameters:
        Xsplit (int): Number of subdomains to be split in the x-axis
        Ysplit (int): Number of subdomains to be split in the y-axis
        Zsplit (int): Number of subdomains to be split in the z-axis
        feat (bool): Whether or not to include color data in the point cloud
        point_path (str): Path to the .npy file containing the point cloud
        label_path (str): Path to the .npy file containing the labels
        color_path (str): Path to the .npy file containing the color data

        Returns:
        batches (list): A list containing dictionaries of point cloud data
        """
        if point_path == None:
            point = np.load(self.file_path_npy)[:,0:3]
            print(f"\nUsing points from {self.file_path_npy} directory")
        else:
            point = np.load(point_path)
                            
        
        if label_path == None:
            label = np.zeros(np.shape(point)[0], dtype = np.int32)
        else:
            label = np.load(label_path)
            
                       
        if(color_path == None and feat == True):  
            color = np.load(self.file_path_npy)[:,3:]
            print(f"Using colors from {self.file_path_npy} directory\n")
        elif(color_path != None and feat == True):
            color = np.load(color_path)
                    
                 
        xmax = max(point[:,0])
        ymax = max(point[:,1])
        zmax = max(point[:,2])
        xmin = min(point[:,0])
        ymin = min(point[:,1])
        zmin = min(point[:,2])

        dom_max = [xmax, ymax, zmax]
        dom_min = [xmin, ymin, zmin]

        split = [Xsplit,Ysplit,Zsplit]

        #Create a boundary limit for each sectional domain using:
        dom_lim = []
        for i in range(len(split)):
            box = list(np.linspace(math.floor(dom_min[i]), 
                                    math.floor(dom_max[i]), 
                                    split[i]+1))
            
            if box[-1] < dom_max[i]:
                box[-1] = int(math.ceil(dom_max[i] +1))

            dom_lim.append(box[1:])

        Xlim,Ylim,Zlim = np.meshgrid(dom_lim[0],dom_lim[1],dom_lim[2])
        Xlim = Xlim.flatten()
        Ylim = Ylim.flatten()
        Zlim = Zlim.flatten()
        Class_limits = np.vstack((Xlim,Ylim,Zlim)).T
        #print(Class_limits)

        
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
                    continue
                
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
    
    
    def CreatePipeline(self,ckpt_path = None):
        
        if ckpt_path == None:
            print("\nFetching checkpoint model in the current directory")
            ckpts = sorted(self.dir.glob("*.pth"))
            ckpt_path = ckpts[-1] #If there is more than one, select the last one in the list of array
            print(f"Using checkpoint model {ckpt_path} to run inference")
              
                        
        print('\nConfiguring model...')    
        try: 
            model = ml3d.models.RandLANet(**self.cfg.model)
            print("RandLANet model configured...")
        except:
            model = ml3d.models.KPFCNN(**self.cfg.model)
            print("KPConv model configured...")
            
           
        pipeline = ml3d.pipelines.SemanticSegmentation(model=model, device="gpu", **self.cfg.pipeline)
        print(f"The device is currently running on: {pipeline.device}\n")
        pipeline.load_ckpt(ckpt_path=ckpt_path)
        
        return pipeline
    
    
    def CustomInference(self,pipeline,batches):
        #Labels is replaced with classification to avoid conflicts
        #once training is considered.
        
        """
        This function takes in a configured pipeline and a list of batches and runs
        inference on the batches. The output is a list of dictionaries, where each
        dictionary contains the name of the point cloud, the points, the prediction
        as a classification name, and the prediction as a label number (model specific).

        Parameters
        ----------
        pipeline : ml3d.pipeline.SemanticSegmentation
            Configured pipeline to run inference with
        batches : list
            List of batches to run inference on

        Returns
        -------
        list
            List of dictionaries, where each dictionary contains the name of the
            point cloud, the points, the prediction as a classification, and the
            prediction as a label
        """
        Results = []
        cfg_name = self.cfg.dataset['name']

        print(f"Name of the dataset used: {cfg_name}") 
        Liststr = f'ml3d.datasets.{cfg_name}.get_label_to_names()'
        Listname = eval(Liststr)     
        key_list = np.array(list(Listname.keys())) 
        val_list = list(Listname.values())
                                                      
        print('Running Inference...')

        for i,batch in enumerate(batches):
            i += 1
            classification = []
            print(f"\nIteration number: {i}")
            results = pipeline.run_inference(batch)
            print('Inference processed successfully...')
            print(f"\nInitial result: {results['predict_labels'][:13]}")
            pred_label = (results['predict_labels'] +1).astype(np.int32) 
            
            for pred in pred_label:
                ind = np.nonzero(pred == key_list)[0]
                ind = ind[0]
                classification.append(val_list[ind])
                
            
            vis_d = {
                "name": batch['name'],
                "points": batch['point'].astype(np.float32),
                "classification": classification,
                "pred": pred_label,
                    }
                               
            pred_val = vis_d["pred"][:13]
            print(f"\nPrediction values: {pred_val}")
            Results.append(vis_d)      
                
        return Results
        
    
    def _Saveto(self,stat,Predictions,interval,Dict_num,maxpoints):
        Data = []
        counter = 1
        Num_point = 0
        try:       
            for Prediction in Predictions:
                catch_err = Prediction["points"][:2,:] #Dummy to catch error
                
                if len(Prediction["points"]) < interval:
                    save = 1
                    Prediction["points"] = (Prediction["points"]).tolist()
                    #Prediction["classfication"] = (Prediction["labels"]).tolist()
                    Prediction["pred"] = (Prediction["pred"]).tolist()
                    Num_point += len(Prediction["pred"])
                    Data.append(Prediction)
                    print(f"\nPoint cloud - {Prediction['name']} has been converted to {stat}")
                    print(f"Length of data: {len(Data)}")
                    print(f"Total number of points saved: {Num_point}")
                
                else:
                    print(f"\nPoint cloud - {Prediction['name']} has exceeded {interval} points\nSplitting the batch...")
                    Range = list(range(0,len(Prediction["points"]),interval))
                    save = 0
                    
                    for i, content in enumerate(Range):
                        if content == Range[-1]:
                            split_name = Prediction["name"] + str('-') + str(i)
                            split_points = Prediction["points"][content:, :]
                            split_labels = Prediction["classification"][content:]
                            split_pred = Prediction["pred"][content:]
                        else:
                            split_name = Prediction["name"] + str('-') + str(i)
                            split_points = Prediction["points"][content:Range[i+1], :]
                            split_labels = Prediction["classification"][content:Range[i+1]]
                            split_pred = Prediction["pred"][content:Range[i+1]]

                        Num_point += len(split_pred)
                        Data.append({
                            "name": split_name,
                            "points": split_points.tolist(),
                            "classification": split_labels,
                            "pred": split_pred.tolist(),
                        })
                        print(f"\nPoint cloud - {split_name} has been converted to {stat}")
                        print(f"Length of data: {len(Data)}")
                        print(f"Total number of points saved: {Num_point}")
                        
                save = 1
                
                if (len(Data) > Dict_num and save == 1) or (Num_point > maxpoints and save == 1):
                    if stat == 'Json':
                        file_name = str(self.folder.stem) + '-' + str(counter) + '-Prediction.json'
                        file_path = self.folder / "split" / file_name
                        with open(file_path, "w") as file:
                            json.dump(Data, file, indent=4)
                    else:
                        file_name = str(self.folder.stem) + '-' + str(counter) + '-Prediction.pkl'
                        file_path = self.folder / "split" / file_name
                        with open(file_path, "wb") as file:
                            pickle.dump(Data, file)
                            
                    print(f"{file_name} has been created and written.")
                    counter += 1
                    Data = []
                    Num_point = 0
        
        except:
            print('\nArray to list conversion is not required. Saving data to a single file...')
            if stat == 'Json':
                file_name = str(self.folder.stem) + '_' + str(counter) + '_Prediction.json'
                file_path = self.folder / file_name
                with open(str(file_path), "w") as file:
                    json.dump(Predictions, file, indent=4)
            else:
                file_name = str(self.folder.stem) + '_' + str(counter) + '_Prediction.pkl'
                file_path = self.folder / file_name
                with open(str(file_path), "wb") as file:
                    pickle.dump(Predictions, file)
            
        
        
    def SavetoJson(self,Predictions,interval = 300000,Dict_num = 14,maxpoints = 1200000):   #Predictions variable refers to Results
        stat = 'Json'
        return self._Saveto(stat,Predictions,interval,Dict_num,maxpoints)
                
                            
    def SavetoPkl(self,Predictions,interval = 300000,Dict_num = 14,maxpoints = 1200000):   #Predictions variable refers to Results
        stat = 'Pickle'
        return self._Saveto(stat,Predictions,interval,Dict_num,maxpoints)
    
    
    def load_data (self,ext,dir_path = None):
        """
        Loads saved result files from a directory specified by the user. If no directory is specified, the function will search for files in the current directory. The function is able to read .pkl and .json files.

        Parameters
        ----------

        ext : str
            The extension of the file to be read

        dir_path : str, optional
            The path to the directory containing the files to be read

        Returns
        -------

        Data : list
            A list of dictionaries containing the point cloud data

        """
        
        if dir_path == None:
            file_name = Path(str(self.folder.stem)+'_1'+f'_Prediction.{ext}')
            singleFile = self.folder / file_name
            #print(f"file_name : {str(file_name)}")
            #print(f"single File path : {str(singleFile)}")
            if singleFile.exists():
                print(f"\nLoading single .{ext} file from {str(singleFile)}")
                Files = [singleFile]
            else:
                split_path = self.folder / "split"
                Files = sorted(split_path.glob(f"*.{ext}"))
        
                if not Files == []:
                    print(f"\nLoading .{ext} files from {split_path}")
                else:
                    print('Error: No related files found in the directory')
                    quit()
        else:            
            Files = sorted(Path(dir_path).glob(f"*.{ext}"))

            print(f"\nLoading .{ext} files from {dir_path}")
        
        
        Data = []
        print("Loading data...")
        if ext == 'pkl':     
            for Filename in Files:
                            
                with open(Filename, 'rb') as file:
                    File = pickle.load(file)
                                            
                for Dicts in File:
                    Data.append(Dicts)
        
        else:
                          
            for Filename in Files:
                            
                with open(Filename, 'r') as file:
                    File = json.load(file)
                                            
                for Dicts in File:
                    Data.append(Dicts)
        return Data

    def _Visualizer(self,ext,dir_path = None): 
                
        cfg_name = self.cfg.dataset['name']
        print(f"Name of the dataset used: {cfg_name}") 
        Liststr = f'ml3d.datasets.{cfg_name}.get_label_to_names()'
        Vis_label = eval(Liststr)
        v = ml3d.vis.Visualizer()
        lut = ml3d.vis.LabelLUT()
        for val in sorted(Vis_label.keys()):
            lut.add_label(Vis_label[val], val)
        v.set_lut("labels", lut)
        v.set_lut("pred", lut)
                   
        Data = self.load_data(ext,dir_path)
        
        for a_Data in Data:
            a_Data.pop("classification")
            
        print('Visualising...') 
        v.visualize(Data)
        
    
    def PklVisualizer(self,dir_path = None):
        ext = 'pkl'
        return self._Visualizer(ext,dir_path)
    
    
    def JsonVisualizer(self,dir_path = None):
        ext = 'json'
        return self._Visualizer(ext,dir_path)
    
    def SavetoLas(self, results, dir_path = None): 
                
        print("Saving point cloud in LAS format with segementation results...")
        if dir_path is None:
            dir_path = self.folder / "lasClassified"
        else: 
            dir_path = Path(dir_path)
                                  
        if not dir_path.exists():
            os.makedirs(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Folder {dir_path} created.")
        else:
            print(f"Folder {dir_path} already exists. overwriting file inside...")
            #delete all files inside the directory
            for filename in dir_path.iterdir():
                file_path = os.path.join(str(dir_path), filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        #get the class names
        cfg_name = self.cfg.dataset['name']
        Liststr = f'ml3d.datasets.{cfg_name}.get_label_to_names()'
        Listname = eval(Liststr)

        #convert the dictionaries to one dataframe
        df = pd.DataFrame()
        for result in results:
            temp_points = pd.DataFrame(result["points"], columns=["x", "y", "z"])
            temp_pred = pd.DataFrame(result['pred'], columns=["pred"])
            temp_df = pd.concat([temp_points, temp_pred], axis=1)
            df = pd.concat([df, temp_df], ignore_index=True)

            #group the points by pred value
        groups = df.groupby("pred")
        for name, group in groups:
            # new_las = laspy.LasData(las.header)
            # new_las.points[las.classification==1].copy()

            header = laspy.LasHeader(point_format=0, version="1.4")
            new_las = laspy.LasData(header)

            np_arr = group.to_numpy()
            new_las.x = np_arr[:,0]
            new_las.y = np_arr[:,1]
            new_las.z = np_arr[:,2]
            filename = str(name) + '_' + Listname[name]+'.las'
            
            new_las.write(dir_path/filename)
        

                    
            
      
        
            