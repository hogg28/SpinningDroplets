# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:21:09 2023

@author: johna
"""

print('hi')
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
#2.5 um/px


### Import image sequence ###
print('hi')

mpl.rc('image', cmap='gray')
plt.ioff()

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

@pims.pipeline
def preprocess_img(frame):
    
    # frame = frame.copy()
    frame = gaussian_filter(frame, sigma=4)
    img = frame

    img = img[:,50:1150]
    img = np.rot90(img)
    img = np.rot90(img)
    img = img[50:800,:]
    
    return img

def find_lens_edge(x, xc, yc, r):
    return np.sqrt(r**2-(x-xc)**2)+yc

def find_edges(frame, idx):
    edge_results = pd.DataFrame(columns=['peak','x_val','frame'])
    for x in range(0,len(frame[0])):
        pos = frame[:,x]
        c=10
        avg_pos = pos
        slope = []

        for i in range(len(avg_pos)-c):
            slope.append((avg_pos[i]-avg_pos[(i+c)])/(c))
        # peak = signal.find_peaks(-np.array(slope), height = 0.8)[0]
        peak = signal.find_peaks(-np.array(slope), prominence = 0.04)[0]
        
        if peak.min() > 30:
            peak = peak.min()
        else:
            peak = peak[1]

        if peak.size == 0:
            peak = np.array([-c])
        

        edge_results = edge_results.append([{'peak': peak.min() +c,
                                              'x_val': x,
                                              'frame': idx,
                                              },])
    return edge_results

class ComputeCurvature:
    def __init__(self):
        """ Initialize some variables """
        self.xc = 500  # X-coordinate of circle center
        self.yc = -1000# Y-coordinate of circle center
        self.r = 7780/2.5   # Radius of the circle
        self.xx = np.array([])  # Data points
        self.yy = np.array([])  # Data points

    def calc_r(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.xx-xc)**2 + (self.yy-yc)**2)

    def f(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        ri = self.calc_r(*c)
        return ri - ri.mean()

    def df(self, c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df_dc = np.empty((len(c), self.xx.size))

        ri = self.calc_r(xc, yc)
        # ri = self.r
        
        df_dc[0] = (xc - self.xx)/ri                   # dR/dxc
        df_dc[1] = (yc - self.yy)/ri                   # dR/dyc
        df_dc = df_dc - df_dc.mean(axis=1)[:, np.newaxis]
        return df_dc

    def fit(self, xx, yy):
        self.xx = xx
        self.yy = yy
        center_estimate = np.r_[np.mean(xx), np.mean(yy)]
        center = optimize.leastsq(self.f, center_estimate, Dfun=self.df, col_deriv=True)[0]

        self.xc, self.yc = center
        ri = self.calc_r(*center)
        self.r = ri.mean()
        return 1 / self.r, self.xc, self.yc, self.r  # Return the curvature
    
# Function to calculate the slope
def normal_slope(a, b, x1, y1):
   
    # Store the coordinates
    # the center of the circle
    g = a / 2
    f = b / 2
 
    # If slope becomes infinity
    if (g - x1 == 0):
        return (-1)
 
    # Stores the slope
    slope = (f - y1) / (g - x1)
 
    # If slope is zero
    if (slope == 0):
        return (-2)
 
    # Return the result
    return slope
 
# Function to find the equation of the
# normal to a circle from a given point
def normal_equation(a, b, x1, y1):
 
    # Stores the slope of the normal
    slope = normal_slope(a, b, x1, y1)
 
    # If slope becomes infinity
    if (slope == -1) :
        print("x = ", x1)
 
    # If slope is zero
    if (slope == -2) :
        print("y = ", y1)
 
    # Otherwise, print the
    # equation of the normal
    if (slope != -1 and slope != -2):
 
        x1 *= -slope
        x1 += y1
 
    return(slope, x1)

#%%

dir_list = pd.read_csv('F:/Johnathan/SpinningDrops/2SDS/08022023/30minAccumulation_large/DirectoryList.csv').directory
list_of_directories = pd.read_csv('F:/Johnathan/SpinningDrops/ListofDirectories_2.csv').directory
prefix = '/*.tiff'

left_edge = 1000
right_edge = 50

# rotation_data = pd.read_csv('F:/Angela/28102022/Data_2.csv')
rotation_data = pd.DataFrame()

#%%
import math

def rotate_math(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def find_surface_area(frame,idx):
    print(idx)
    mpl.rc('image', cmap='gray')
    plt.ioff()
    warnings.simplefilter(action='ignore', category=FutureWarning)
    global Aggregate_structure
    Aggregate_structure = pd.DataFrame()
    
    # thresh = threshold_otsu(frame)
    # block_size = 301
    # local_thresh = threshold_local(frame, block_size, offset=10)
    # binary = frame > local_thresh
    # binary = frame > thresh
    binary = frame > 150
    
    #Find edges of incorreclty oriented image
    edges = find_edges(binary,idx)
    
    #Find the edges of the lens and fit a circle
    lens_edge = edges[((edges.x_val<50) | (edges.x_val>1050))]
    x = np.r_[pd.to_numeric(lens_edge.x_val, errors='coerce')]
    y = np.r_[pd.to_numeric(lens_edge.peak, errors='coerce')]
    comp_curv = ComputeCurvature()
    curvature, xc, yc, r = comp_curv.fit(x, y)
    print(curvature, yc)

    
    ''' if the curvature isn't good need to change the values for lens edges'''
    i = -1
    while r > 5500 or r < 4000 or yc > 0:
        i+=1
        print(i)
        lens_edge = edges[(edges.x_val>(0+i)) & (edges.x_val<(50+i)) | ((edges.x_val>(len(frame[0])-(50+i))) & (edges.x_val<(len(frame[0])-i)))]
        # lens_edge = edges[(((edges.x_val>(0+i)) & (edges.x_val<(50+i))) | (edges.x_val>(len(frame[0])-(50+i)) & (edges.x_val<(len(frame[0])-i))))]
        if i > 100:
            lens_edge = edges[(edges.x_val>(0+(i-100))) & (edges.x_val<(25+(i-100))) | ((edges.x_val>(len(frame[0])-(25+(i-100)))) & (edges.x_val<(len(frame[0])-(i-100))))]
        x = np.r_[pd.to_numeric(lens_edge.x_val, errors='coerce')]
        y = np.r_[pd.to_numeric(lens_edge.peak, errors='coerce')]
        comp_curv = ComputeCurvature()
        curvature, xc, yc, r = comp_curv.fit(x, y)
        print(r, yc)
    
    #Plot and save figure of initial edge detection
    plt.figure()
    plt.imshow(frame)
    plt.plot(edges.x_val, edges.peak, lw =0.5, color = 'blue')
    theta_fit = np.linspace(-np.pi, np.pi, 180)
    x_fit = comp_curv.xc + comp_curv.r*np.cos(theta_fit)
    y_fit = comp_curv.yc + comp_curv.r*np.sin(theta_fit)
    plt.plot(x_fit, y_fit, 'r--', label='fit', lw=2)
    plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
    plt.xlim([-500,1500])
    plt.ylim([-50,600])
    plt.xlabel('x')
    plt.ylabel('y')
    

    plt.savefig(directory+ '/initial_edges_4/'+str(idx)+'.png')
    plt.close()
    
    y = int(edges[edges.x_val==500].peak) -50
    # binary = flood_fill(binary, (500, y), 0)

    
    edges['y_lens'] = np.array(edges.groupby('x_val').apply(lambda x: find_lens_edge(x.name,xc,yc,r)))
    
    for i in range(len(binary[0])):
        for j in edges[edges.x_val==i].y_lens:
            binary[:int(j), i] = True
            
    # binary = binary_closing(binary)
    binary = binary_closing(binary,iterations = 3)
    binary = np.invert(binary)
    label_im, nb_labels = ndimage.label(binary)
    sizes = ndimage.sum(binary, label_im, range(nb_labels + 1))
    mask = sizes > 20000
    binary = mask[label_im]
    binary = np.invert(binary)
    # binary = flood_fill(binary)
    # frame = frame[40:460,40:1150]
    
    # frame = frame[np.where(binary==False)[0],np.where(binary==False)[0]]
    masked_frame = frame
    masked_frame[~(binary)]=0

    
    plt.figure()
    plt.imshow(masked_frame)
    plt.savefig(directory+ '/binary_4/'+str(idx)+'.png')
    plt.close()
    
    binary = binary[20:650,20:1050]
    
    x_left = np.where(binary == False)[1].min()
    x_right = np.where(binary == False)[1].max()
    
    y_left = np.where(binary == False)[0][np.where(binary == False)[1]==x_left][0]
    y_right = np.where(binary == False)[0][np.where(binary == False)[1]==x_right][0]

    
    m = (y_right-y_left)/(int(x_right)-int(x_left))
    
    print(m, np.arctan(-m))
    
    angle = np.degrees(np.arctan(m))

                 
    frame = rotate(frame, axes = (1,0), angle=angle, reshape = False)
    
    left, top = rotate_math([len(frame[0])/2, len(frame)/2], [len(frame[0]), 0], angle)
    right, bottom = rotate_math([len(frame[0])/2, len(frame)/2], [0, len(frame)], angle)
    # right, top = rotate_math([0,0], [len(frame), len(frame[0])], angle)
    
    mask = frame > 0
    
    mask = mask[75:int(len(frame)-75),50:(len(frame[0])-50)]
    
    mask = np.invert(mask)
    label_im, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_2 = sizes > 20000
    mask = mask_2[label_im]
    mask = np.invert(mask)
    
    plt.figure()
    plt.imshow(mask)
    plt.savefig(directory+ '/rotated_image_4/'+str(idx)+'.png')
    plt.close()
    
    D = []
    j=0
    for i in range(len(mask)):
        
        if  len(np.where(mask[i,:]==0)[0])==0:
            continue
        else:
            j+=1
            # print(np.where(mask[i,:]==0), np.where(mask[i,:]==0)[0])
            x_left = np.where(mask[i,:]==0)[0].min()
            x_right = np.where(mask[i,:]==0)[0].max()
            if j ==10:
                bottom = i
                width = x_right-x_left
                j =0
            top = i
        
        plt.plot([x_left,x_right], [i,i])
        
        D.append(x_right-x_left)
    height = top-bottom+10
        
    plt.ylim([25,450])
    plt.xlim([0,1000])
    plt.savefig(directory+ '/aggregate_structure_4/'+str(idx)+'.png')
    plt.close()
    
    D = np.array(D)

    circumfrence = np.pi*D
    volume = np.pi*(D/2)**2
    
    circumfrence = circumfrence[~np.isnan(circumfrence)]
    volume = volume[~np.isnan(volume)]
    
    surface_area = np.sum(circumfrence)
    volume = np.sum(volume)
    
    Aggregate_structure = Aggregate_structure.append([{'surface_area': surface_area,
                                                       'volume': volume,
                                                       'height': height,
                                                       'frame': idx,
                                                       'width': width,
                                                       },])
    return Aggregate_structure

trial = 0
for list_of_dir in list_of_directories:
    trial+=1
    rotation_data = pd.DataFrame()
    dir_list = pd.read_csv(list_of_dir).directory
    for directory in dir_list:
        data = pd.DataFrame()
        frames = preprocess_img(pims.ImageSequence(os.path.join(directory+prefix)))
        print(directory)
                
        if not os.path.exists(directory+'/initial_edges_4'):
            os.mkdir(directory+'/initial_edges_4')
        if not os.path.exists(directory+'/centre_of_mass'):
            os.mkdir(directory+'/centre_of_mass')
        if not os.path.exists(directory+'/rotated_image_4'):
            os.mkdir(directory+'/rotated_image_4')
        if not os.path.exists(directory+'/final_edges'):
            os.mkdir(directory+'/final_edges')
        if not os.path.exists(directory+'/aggregate_structure_4'):
            os.mkdir(directory+'/aggregate_structure_4')
        if not os.path.exists(directory+'/binary_4'):
            os.mkdir(directory+'/binary_4')
        
        # for idx, frame in enumerate(frames):
        #     Aggregate_structure = find_surface_area(frame, idx)
    
        from joblib import Parallel, delayed
        Data = Parallel(n_jobs = 16)(delayed(find_surface_area)(img, num) for num, img in enumerate(frames))
        Aggregate_structure = pd.concat(Data)
        
        filename = directory.split('/')[-1]
        rpm = int(re.search('\d+rpm',filename).group(0).split('rpm')[0])
        ramp = directory.split('/')[-2].split('_')[-1]
        frame_rate = float(re.search('\d+.\d+Hz',filename).group(0).split('Hz')[0])
        rotation_speed = rpm/60
        Aggregate_structure.to_csv(directory+'/aggregate_structure.csv')
        
        data['height'] = [Aggregate_structure['height'].mean()]
        data['height_std'] = [Aggregate_structure['height'].std()]
        data['width'] = [Aggregate_structure['width'].mean()]
        data['width_std'] = [Aggregate_structure['width'].std()]
        data['SA'] = [Aggregate_structure['surface_area'].mean()]
        data['SA_std'] = [Aggregate_structure['surface_area'].std()]
        data['vol'] = [Aggregate_structure['volume'].mean()]
        data['vol_std'] = [Aggregate_structure['volume'].std()]
        data['ramp'] = [ramp]
        data['loop'] = ['a']
        data['rotation_speed'] = [rotation_speed]
        data['frame_rate'] = [frame_rate]
        data['filename'] = [filename]
        data['SDS'] = [(directory.split('/')[3]).split('SDS')[0]]
        data['Trial'] =[str(trial)]
        
        rotation_data = rotation_data.append(data)
    
    text = (directory.split('/')[:-2])
    data_directory = '/'.join(text)+'/Data_5.csv'
    rotation_data.to_csv(data_directory)