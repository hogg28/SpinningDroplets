# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:55:03 2023

@author: johna
"""

### I will manually remove outlier data which resulted from poor analysis. I will then read in
### the new updated csv and create a new mean data without outliers
import numpy as np
import pims
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
from scipy import optimize
from scipy.ndimage import rotate
import re
from skimage.filters import threshold_otsu, threshold_local, unsharp_mask
from skimage.morphology import flood_fill
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_closing
from scipy import ndimage
from scipy.signal import savgol_filter

#List of master directories
list_of_directories = pd.read_csv('C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/ListofDirectories.csv').directory

for list_of_dir in list_of_directories:
    rotation_data = pd.DataFrame()
    test = list_of_dir.split('/')[2:]
    list_of_dir = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/'+'/'.join(test)
    dir_list = pd.read_csv(list_of_dir).directory
    for directory in dir_list:
        data = pd.DataFrame()
        test2 = directory.split('/')[2:]
        directory = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/'+'/'.join(test2)
        
        filename = directory.split('/')[-1]
        rpm = int(re.search('\d+rpm',filename).group(0).split('rpm')[0])
        ramp = directory.split('/')[-2].split('_')[-1]
        frame_rate = float(re.search('\d+.\d+Hz',filename).group(0).split('Hz')[0])
        rotation_speed = rpm/60
        try:
            Aggregate_structure = pd.read_csv(directory + '/aggregate_structure_1.csv')
            print('IT WORKED')
            # df = Aggregate_structure
            # Aggregate_structure = df[pd.qcut(df['size'],q=4,labels=[1,2,3,4], duplicates='raise')<3]
        except:
            Aggregate_structure = pd.read_csv(directory + '/aggregate_structure.csv')
            df = Aggregate_structure
            Aggregate_structure = df[pd.qcut(df['size'],q=4,labels=[1,2,3,4], duplicates='raise')<3]
        data['height'] = [Aggregate_structure['height'].mean()]
        data['height_std'] = [Aggregate_structure['height'].std()]
        data['width'] = [Aggregate_structure['width'].mean()]
        data['width_std'] = [Aggregate_structure['width'].std()]
        data['size'] = [Aggregate_structure['size'].mean()]
        data['size_std'] = [Aggregate_structure['size'].std()]
        data['SA'] = [Aggregate_structure['surface_area'].mean()]
        data['SA_std'] = [Aggregate_structure['surface_area'].std()]
        data['vol'] = [Aggregate_structure['volume'].mean()]
        data['vol_std'] = [Aggregate_structure['volume'].std()]
        data['ramp'] = [ramp]
        data['loop'] = ['a']
        data['rotation_speed'] = [rotation_speed]
        data['frame_rate'] = [frame_rate]
        data['filename'] = [filename]
        data['SDS'] = [(directory.split('/')[8]).split('SDS')[0]]
        data['Trial'] =[str(1)]
        
        rotation_data = rotation_data.append(data)
    text = (directory.split('/')[:-2])
    data_directory = '/'.join(text)+'/Data_8.csv'
    
    rotation_data.to_csv(data_directory)
    print(directory)