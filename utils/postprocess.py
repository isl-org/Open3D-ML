import numpy as np
import os
import sys
import math
from pathlib import Path
from typing import Literal

class PostProcess():
    def __init__(self) -> None:
        
        main_path = os.path.abspath(sys.argv[0])
        self.dir = Path(main_path).parent
        self.file_path = self.dir / "Sliced_points.npy"
    
    def pc_slice(self,Start,End,tolerance,result,plane: Literal['XY', 'XZ', 'YZ'] = 'XY'):
        #Straight line function of y = mx + c
        
        #change and res would refer to the manipulated and responding variables
        #which vary depending on the chosen plane
        print(f"\nDomain range of X-axis: [{min(result[:,0])} {max(result[:,0])}]")
        print(f"Domain range of Y-axis: [{min(result[:,1])} {max(result[:,1])}]")
        print(f"Domain range of Z-axis: [{min(result[:,2])} {max(result[:,2])}]")
        
        change_ind, res_ind = self.planar_axes(plane)
        top_dif = End[res_ind] - Start[res_ind]
        bottom_dif = End[change_ind] - Start[change_ind]
        sample_points = result[:,change_ind]
        
        if top_dif > 1000*bottom_dif: #by passsing divisional problem when m is approaching to infinity (vertical straight line)
            prop = [Start[change_ind]]
        else:
            m = top_dif/bottom_dif
            c = End[res_ind] - m*End[change_ind]
            prop = [m,c]
        
        benchmark = self.tolerance_limit(prop,tolerance)
        print(f"Benchmark value: {benchmark}") #should return a single value
        
        #when straight line is not vertical
        if not top_dif > 1000*bottom_dif:
            
            res_points = self.create_line(m,c,sample_points)
            vertical_dif = abs(result[:,res_ind] - res_points)
            index = np.asarray(benchmark>vertical_dif).nonzero()
            sliced_points = result[index]
                        
            #Filtering the points based on the length of the drawn line
            change_condition = np.logical_and(Start[change_ind] < sliced_points[:,change_ind],End[change_ind] > sliced_points[:,change_ind])
            index_change = np.asarray(change_condition).nonzero()
            sliced_points = sliced_points[index_change]
            
            if m > 0.11: #threshold when m is approaching to 0
                res_condition = np.logical_and(Start[res_ind] < sliced_points[:,res_ind],End[res_ind] > sliced_points[:,res_ind])
                index_res = np.asarray(res_condition).nonzero()
                sliced_points = sliced_points[index_res]
            
            print(f"Length of sliced data: {len(sliced_points)}")
            
        else:
                       
            low_lim = prop[0]  - benchmark #Start[change_ind] - tolerance
            high_lim = prop[0]  + benchmark #Start[change_ind] + tolerance
            condition = np.logical_and(low_lim< result[:,change_ind], high_lim > result[:,change_ind])
            index = np.asarray(condition).nonzero()
            sliced_points = result[index]
            
            print(f"NOTE: Vertical line in {plane}-plane detected")
            print(f"Length of sliced data: {len(sliced_points)}")
                   
        return np.save(str(self.file_path),sliced_points)
        
    def tolerance_limit(self,prop,tolerance): 
        #tolerance is defined as the perpendicular distance from an imaginary offset line to the created line
        
        if len(prop) == 1: #for vertical straight line
            compare = tolerance
        else:
            m = abs(prop[0]) #converting the value to be absolute
            slope = math.atan(m) #slope angle of the line in rads
            vertical_dist = tolerance/ math.cos(slope) #vertical distance of original line 
            #to the offset line using the same x coordinate value (changing variable)   
            compare = vertical_dist
            
        return compare  
    
    
    def planar_axes(self,plane):
        #return index of the two axes that make up the plane
        #array is expected to be in [x,y,z] orientation
        #index is based on manipulated(change) and responding(res) variables
        
        if plane == 'XY': #plan view
            change_ind = 0 #X
            res_ind = 1 #Y
       
        elif plane == 'XZ': #front view
            change_ind = 0 #X
            res_ind = 2    #Z
        
        elif plane == 'YZ': #side view
            change_ind = 1 #Y
            res_ind = 2 #Z
        else:
            print('\nError: Plane is not correctly define')
            quit()
                
        return change_ind, res_ind
    
    
    def create_line(self,m,c,sample_points):
                
        res_points = m*sample_points + c
               
        return res_points     
        
        
        
        

