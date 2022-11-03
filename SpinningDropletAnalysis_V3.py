# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:29:51 2022

@author: johna
"""

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
from scipy.ndimage.interpolation import rotate
import re
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import interp1d
#2.5 um/px

### Import image sequence ###

mpl.rc('image', cmap='gray')

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def auto_rotate(frame):
    c=5
    left_side = frame[100:1200,100]
    left_side = moving_average(left_side, n=5)
    right_side = frame[100:1200,1100]
    right_side = moving_average(right_side, n=5)
    left_slope = []
    right_slope = []
    for i in range(len(left_side)-c):
        left_slope.append((left_side[i+c]-left_side[i])/(c))
    for i in range(len(right_side)-c):
        right_slope.append((right_side[i+c]-right_side[i])/(c))

    #find peak on both left and right
    left_peak = signal.find_peaks(np.array(left_slope), prominence = 2.5)[0]
    right_peak = signal.find_peaks(np.array(right_slope), prominence = 2.5)[0]
    while np.abs(left_peak[0] - right_peak[0])>3:
        angle = 1
        left_side = frame[100:1200,100]
        left_side = moving_average(left_side, n=5)
        right_side = frame[100:1200,1100]
        right_side = moving_average(right_side, n=5)
        left_slope = []
        right_slope = []
        for i in range(len(left_side)-c):
            left_slope.append((left_side[i+c]-left_side[i])/(c))
        for i in range(len(right_side)-c):
            right_slope.append((right_side[i+c]-right_side[i])/(c))
        left_peak = signal.find_peaks(np.array(left_slope), prominence = 1.5)[0]
        right_peak = signal.find_peaks(np.array(right_slope), prominence = 1.5)[0]
        if np.abs((left_peak[0] - right_peak[0]))<15:
            angle = 0.1
        if left_peak[0] - right_peak[0] > 0:
            frame = rotate(frame, angle=-angle)
        elif left_peak[0] - right_peak[0] < 0:
            frame = rotate(frame, angle=angle)
    return(frame)



@pims.pipeline
def preprocess_img(frame):
    
    frame = np.rot90(frame)
    frame = np.rot90(frame)

    frame= auto_rotate(frame)
    frame = frame[100:700,100:1150]
    return frame

def find_edges(frame, idx):
    edge_results = pd.DataFrame(columns=['peak','x_val','frame'])
    for x in range(0,len(frame[0])):
        pos = frame[:,x]
        c=20
        avg_pos = moving_average(pos, n=c)
        slope = []

        for i in range(len(avg_pos)-c):
            slope.append((avg_pos[i]-avg_pos[i+c])/(c))
        peak = signal.find_peaks(-np.array(slope), prominence = 2.5)[0]

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

dir_list = pd.read_csv('D:/Angela/28102022/DirectoryList.csv').directory
prefix = '*.tiff'



# rotation_data = pd.read_csv('F:/Johnathan/SpinningDrops/092022/30092022/2SDS/2SDSlarge_170.21Hz/Data.csv')
rotation_data = pd.DataFrame()




for directory in dir_list:
    # directory = 'D:'+(directory.split(':')[1])
    data = pd.DataFrame()
    frames = frames = preprocess_img(pims.ImageSequence(os.path.join(directory+prefix)))
    print(directory)
    Aggregate_structure = pd.DataFrame()
    for idx, frame in enumerate(frames[:]):
        #Find edges of incorreclty oriented image
        edges = find_edges(frame,idx)
        
        #Find the edges of the lens and fit a circle
        lens_edge = edges[((edges.x_val<100) | (edges.x_val>1000))]
        x = np.r_[pd.to_numeric(lens_edge.x_val, errors='coerce')]
        y = np.r_[pd.to_numeric(lens_edge.peak, errors='coerce')]
        comp_curv = ComputeCurvature()
        curvature, xc, yc, r = comp_curv.fit(x, y)
        
        #Use circle to rotate image such that rotation axis is pointing upwards
        agg_edge = edges
        agg_edge.groupby(['frame','x_val']).apply(lambda x: x.x_val-x.peak )
        shortest_distance = []
        for x1 in agg_edge.x_val:
            y1 =float(agg_edge[agg_edge.x_val ==x1].peak)
            shortest_distance.append(np.abs(np.sqrt((x1-xc)**2 +(y1-yc)**2)-r))
            
        #Use centre of mass to rotate image
        get_com = lambda m: np.round(np.sum(np.arange(m.shape[0])*m)/np.sum(m))
        com = get_com(np.array(shortest_distance))

        x1=com
        y1 = agg_edge[agg_edge.x_val == com].peak

        a = 2*xc
        b = 2*yc
        c = xc**2+yc**2-r**2
        x1 = float(agg_edge.x_val.iloc[int(com)])
        y1 = float(agg_edge.peak.iloc[int(com)])     
        
     
        # Function Call
        slope, y_int = normal_equation(a, b, x1, y1)
        
        frame = rotate(frame, angle=-np.arctan(slope))
        frame = frame[:,50:1050]
        
        # plt.figure()
        # plt.imshow(frames[idx])
        # theta_fit = np.linspace(-np.pi, np.pi, 180)
        # x_fit = comp_curv.xc + comp_curv.r*np.cos(theta_fit)
        # y_fit = comp_curv.yc + comp_curv.r*np.sin(theta_fit)
        # plt.plot(x_fit, y_fit, 'k--', label='fit', lw=2, color = 'red')
        # plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
        # plt.xlim([-500,1500])
        # plt.ylim([-400,400])
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('curvature = {:.3e}'.format(curvature))
        
        # plt.show()
        
        
        #Detect edges again in the properly rotated frame
        edge_results = find_edges(frame, idx)
        edge_results = edge_results.sort_values(by=['x_val'])
        
        #Find the location of the lens in the properly rotated frame
        lens_edge = edge_results[((edge_results.x_val<100) | (edge_results.x_val>950))]
        x = np.r_[pd.to_numeric(lens_edge.x_val, errors='coerce')]
        y = np.r_[pd.to_numeric(lens_edge.peak, errors='coerce')]
        comp_curv = ComputeCurvature()
        curvature, xc, yc, r = comp_curv.fit(x, y)
        
        # plt.figure()
        # plt.imshow(frame)
        # theta_fit = np.linspace(-np.pi, np.pi, 180)
        # x_fit = comp_curv.xc + comp_curv.r*np.cos(theta_fit)
        # y_fit = comp_curv.yc + comp_curv.r*np.sin(theta_fit)
        # plt.plot(x_fit, y_fit, 'k--', label='fit', lw=2, color = 'red')
        # plt.plot(x, y, 'ro', label='data', ms=8, mec='b', mew=1)
        # plt.xlim([-500,1500])
        # plt.ylim([-400,400])
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title('curvature = {:.3e}'.format(curvature))
        
        # plt.show()
        
        #Subtract off y-value of lens from y-value of peak find height of aggregate
        def find_lens_edge(x, xc, yc, r):
            return np.sqrt(r**2-(x-xc)**2)+yc
        
        edge_results['y_val'] = np.array(edge_results.groupby('x_val').apply(lambda x: x.peak - find_lens_edge(x.name,xc,yc,r)))
        edge_results['y_lens'] = np.array(edge_results.groupby('x_val').apply(lambda x: find_lens_edge(x.name,xc,yc,r)))
        height = edge_results.y_val.max()
        
        agg_edge = edge_results
        agg_edge.groupby(['frame','x_val']).apply(lambda x: x.x_val-x.peak )
        shortest_distance = []
        for x1 in agg_edge.x_val:
            y1 =float(agg_edge[agg_edge.x_val ==x1].peak)
            shortest_distance.append(np.abs(np.sqrt((x1-xc)**2 +(y1-yc)**2)-r))

        com_2 = get_com(np.array(shortest_distance))
        height_com = edge_results[edge_results.x_val==com_2].y_val.max()
        
        
        x_val = edge_results.x_val
        y_val = edge_results.y_val
        f = interp1d(x_val, y_val)
        x_new = np.arange(0, len(frame[0])-1,  0.05)
        y_new = f(x_new)

        interpolated_edges = pd.DataFrame()
        interpolated_edges['x_new'] = x_new
        interpolated_edges['y_new'] = f(x_new)

        #Find the exact shape of the aggregate
        bin_edges = np.arange(2, edge_results.y_val.max(), 1)
        D = []
        for i in range(len(bin_edges)-1):
            y_bottom = bin_edges[i]-0.5
            y_top = bin_edges[i+1]+0.5
            y_mid = (y_top+y_bottom)/2
        
            
            x_left = interpolated_edges[(interpolated_edges.y_new>y_bottom) & (interpolated_edges.y_new<y_top)].x_new.min()
            x_right = interpolated_edges[(interpolated_edges.y_new>y_bottom) & (interpolated_edges.y_new<y_top)].x_new.max()
            
            D.append(x_right-x_left)

        D = np.array(D)
        plt.ylim([0,600])

        circumfrence = np.pi*D
        volume = np.pi*(D/2)**2
        
        circumfrence = circumfrence[~np.isnan(circumfrence)]
        volume = volume[~np.isnan(volume)]
        
        surface_area = np.sum(circumfrence)
        volume = np.sum(volume)
        
        
        Aggregate_structure = Aggregate_structure.append([{'surface_area': surface_area,
                                                           'volume': volume,
                                                           'height': height,
                                                           'height_com': height_com,
                                                           'frame': idx,
                                                           },])
        print(idx)
    
    filename = directory.split('/')[-2]
    rpm = int(re.search('\d+RPM',filename).group(0).split('RPM')[0])
    frame_rate = float(re.search('\d+.\d+Hz',filename).group(0).split('Hz')[0])
    rotation_speed = rpm/60
    
    data['height'] = [Aggregate_structure['height'].mean()]
    data['height_std'] = [Aggregate_structure['height'].std()]
    data['height_com'] = [Aggregate_structure['height_com'].mean()]
    data['height_com_std'] = [Aggregate_structure['height_com'].std()]
    data['SA'] = [Aggregate_structure['surface_area'].mean()]
    data['SA_std'] = [Aggregate_structure['surface_area'].std()]
    data['vol'] = [Aggregate_structure['volume'].mean()]
    data['vol_std'] = [Aggregate_structure['volume'].std()]
    data['rotation_speed'] = [rotation_speed]
    data['frame_rate'] = [frame_rate]
    data['filename'] = [filename]
    
    rotation_data = rotation_data.append(data)
    

    
    
rotation_data.to_csv('D:/Angela/28102022/Data_2.csv')