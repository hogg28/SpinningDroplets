# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:01:46 2022

@author: johna
"""

'''
The goal of this code is to reconstrcut the 3D aggregate of droplets using 
2D edges
'''

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


# directory = 'D:/Angela/28102022/2SDS_initial_4V_3V/3.0V_797RPM_13.1Hz/'
# prefix = '*.tiff'


# plt.figure()
# frames = preprocess_img(pims.ImageSequence(os.path.join(directory+prefix)))


# plt.imshow(frames[20])


#%%
### Detect edges ###


###Bottom to Top###
edge_results = pd.DataFrame(columns=['peak','x_val','frame'])

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
    
# for idx, frame in enumerate(frames[0:1]):
#     edge_results = (find_edges(frame, idx))
#     print(idx)

#%%

### Fit a circle to the lens ###

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


for i in edge_results.frame.unique():
    edges = edge_results[edge_results.frame ==i]
    lens_edge = edges[(edges.frame ==i) & ((edges.x_val<100) | (edges.x_val>1000))]
    x = np.r_[pd.to_numeric(lens_edge.x_val, errors='coerce')]
    y = np.r_[pd.to_numeric(lens_edge.peak, errors='coerce')]
    comp_curv = ComputeCurvature()
    curvature, xc, yc, r = comp_curv.fit(x, y)
    
    # Plot the result
    # plt.figure()
    # plt.imshow(frames[i])
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
    # np.sqrt((self.xx-xc)**2 + (self.yy-yc)**2)
    
#%%
#Find perpendicular lines to the circles defined by the lens

#Realisiticly, I only need to search in region xc-200 to xc+200
# edges = edge_results
agg_edge = edges[(edges.frame ==i) & ((edges.x_val>(xc-200)) & (edges.x_val<(xc+200)))]
agg_edge = edges[(edges.frame ==i)]
agg_edge.groupby(['frame','x_val']).apply(lambda x: x.x_val-x.peak )
shortest_distance = []
for x1 in agg_edge.x_val:
    y1 =float(agg_edge[agg_edge.x_val ==x1].peak)
    shortest_distance.append(np.abs(np.sqrt((x1-xc)**2 +(y1-yc)**2)-r))
    
get_com = lambda m: np.round(np.sum(np.arange(m.shape[0])*m)/np.sum(m))
com = get_com(np.array(shortest_distance))

x1=com
y1 = agg_edge[agg_edge.x_val == com].peak
plt.plot(x1, y1, 'ro')

# np.argmax(shortest_distance)
# x1=np.argmax(shortest_distance)+(xc-200)
# y1 = agg_edge[agg_edge.x_val == np.round(np.argmax(shortest_distance)+(xc-200))].peak
# plt.plot(x1, y1, 'ro')

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
 
# Driver Code
 
# Given Input
a = 2*xc
b = 2*yc
c = xc**2+yc**2-r**2
x1 = float(agg_edge.x_val.iloc[50])
y1 = float(agg_edge.peak.iloc[50])
 
# Function Call
slope, y_int = normal_equation(a, b, x1, y1)

#%%
dir_list = pd.read_csv('F:/Angela/122022/13122022_2SDS/DirectoryList.csv').directory


# rotation_data = pd.read_csv('F:/Johnathan/SpinningDrops/092022/30092022/2SDS/2SDSlarge_170.21Hz/Data.csv')
rotation_data = pd.DataFrame()




for directory in dir_list:
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
    

    
    
rotation_data.to_csv('F:/Angela/122022/13122022_2SDS/Data.csv')
    
    
    
#%%
    
# edges = edge_results.groupby(['frame','peak'])
# D = edges.max()-edges.min() 
# circ = np.pi*D
# vol = np.pi*(D/2)**2

# Aggregate_structure['vol'] = vol.groupby('frame').sum()
# Aggregate_structure['SA'] = circ.groupby('frame').sum()
# Aggregate_structure['frame'] = Aggregate_structure.index

# filename = directory.split('/')[-2]
# rpm = int(re.search('\d+rpm',filename).group(0).split('rpm')[0])
# frame_rate = float(re.search('\d+.\d+ramp',filename).group(0).split('ramp')[0])
# rotation_speed = rpm/60

# # Aggregate_structure.to_csv(directory + 'shapeData.csv')

# data = pd.DataFrame()

# # rotation_data = pd.read_csv('F:/Johnathan/SpinningDrops/092022/30092022/2SDS/2SDSlarge_170.21Hz/Data.csv')
# rotation_data = pd.DataFrame()

# data['height'] = [height.mean()]
# data['SA'] = [Aggregate_structure['SA'].mean()]
# data['SA_std'] = [Aggregate_structure['SA'].std()]
# data['vol'] = [Aggregate_structure['vol'].mean()]
# data['vol_std'] = [Aggregate_structure['vol'].std()]
# data['rotation_speed'] = [rotation_speed]
# data['frame_rate'] = [frame_rate]
# data['filename'] = [filename]

# rotation_data = rotation_data.append(data)


# #I should make everything a function so that it is easy to repeat and easy to batch all the data


# #%%
# plt.figure()
# # plt.plot(edge_results[edge_results.frame ==25].x_val,edge_results[edge_results.frame ==25].peak)
# plt.plot(lens_edge.x_val,lens_edge.peak)

# # plt.plot(edge_results[edge_results.frame ==10].left_peak,edge_results[edge_results.frame ==10].y_val, 'ro')
# # plt.plot(edge_results[edge_results.frame ==10].right_peak,edge_results[edge_results.frame ==10].y_val, 'bo')
# plt.imshow(frames[25])
# # plt.hlines(150, 20, 900, color = 'black', linestyles = 'dashed')
# #%%
# rotate(edge_results)
# #%%

# ### Find the height of the aggregate###

# height = edge_results.groupby('frame').peak.max() -edge_results.groupby('frame').peak.min()
# height = np.array(height)



# ### Find surface area and volume of the aggregate through volume of rotation ###

# Aggregate_structure = pd.DataFrame()

# edges = edge_results.groupby(['frame','peak'])
# D = edges.max()-edges.min() 
# circ = np.pi*D
# vol = np.pi*(D/2)**2

# Aggregate_structure['vol'] = vol.groupby('frame').sum()
# Aggregate_structure['SA'] = circ.groupby('frame').sum()
# Aggregate_structure['frame'] = Aggregate_structure.index

# filename = directory.split('/')[-2]
# rpm = int(re.search('\d+rpm',filename).group(0).split('rpm')[0])
# frame_rate = float(re.search('\d+.\d+Hz',filename).group(0).split('Hz')[0])
# rotation_speed = rpm/60

# # Aggregate_structure.to_csv(directory + 'shapeData.csv')

# data = pd.DataFrame()

# rotation_data = pd.read_csv('F:/Johnathan/SpinningDrops/092022/30092022/2SDS/2SDSlarge_170.21Hz/Data.csv')
# # rotation_data = pd.DataFrame()

# data['height'] = [height.mean()]
# data['SA'] = [Aggregate_structure['SA'].mean()]
# data['SA_std'] = [Aggregate_structure['SA'].std()]
# data['vol'] = [Aggregate_structure['vol'].mean()]
# data['vol_std'] = [Aggregate_structure['vol'].std()]
# data['rotation_speed'] = [rotation_speed]
# data['frame_rate'] = [frame_rate]
# data['filename'] = [filename]

# rotation_data = rotation_data.append(data)

# rotation_data.to_csv('F:/Johnathan/SpinningDrops/092022/30092022/2SDS/2SDSlarge_170.21Hz/Data.csv')

#%%

#I need to save plots of this analysis to confirm it is doing what I want it to do.
#I want to plot the left and right and make it white in the 
# def fill_pipette(frame, edges):
#     filled = np.empty((frame.shape[0],frame.shape[1]))
#     for k in range(frame.shape[0]):
#         for j in range(frame.shape[1]):
#             if frame[k,j] > edges.max & edges.min:
#                 filled[k,j] = 0
#             else:
#                 filled[k,j] = 1
#     return filled

#go from top to bottom, then go left to right
#when you hit a peak value, make it white all the way to the next edge
# for frame in frames[0]:
#     filled = np.empty((frame.shape[0],frame.shape[1]))
#     edges = edge_results[edge_results.frame==frame]
#     for k in range(frame.shape[1]):
#         for j in range(frame.shape[0]):
#             if frame[k,j] = edge_results.peak
#%%
# test = frames[20]
# filled = np.empty((frame.shape[0],frame.shape[1]))
# for p in test.peak.unique():
#     right = test[test.peak==p].x_val.min()
#     left = test[test.peak==p].x_val.max()
#     if test[test.peak==p] is empty:
#         right = test[test.peak==p-1].x_val.min()
#         left = test[test.peak==p-1].x_val.max()
#     filled[p, right:left]=1

# plt.figure()
# plt.imshow(filled)

#%%
# filled = np.empty((frame.shape[0],frame.shape[1]))
# for x in test.x_val.unique():
#     top = test[test.x_val==x].peak.min()
#     bottom = test.peak.min()
#     filled[bottom:top,x]=1

# plt.figure()
# plt.imshow(filled)


#%%
# #use regular expressions to extract frame rate and rotation speed from directory

# rotation_speed = 1674/60
# frame_rate = 27.85
# angle_per_frame = rotation_speed*1/frame_rate*360

# angle_per_frame = 7.2

# import math

# def rotation_matrix(axis, theta):
#     """
#     Return the rotation matrix associated with counterclockwise rotation about
#     the given axis by theta radians.
#     """
#     axis = np.asarray(axis)
#     axis = axis / math.sqrt(np.dot(axis, axis))
#     a = math.cos(theta / 2.0)
#     b, c, d = -axis * math.sin(theta / 2.0)
#     aa, bb, cc, dd = a * a, b * b, c * c, d * d
#     bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
#     return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
#                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
#                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

# def rotate_points(x, y, z, theta):
#     axis = [0,450,0]
#     v = [x, y, z]
    
#     return np.dot(rotation_matrix(axis, theta), v)

# def rotate_img(angle, frame, edge):
#     theta = angle*frame
#     # z = rotate_points(edge.left_peak.iloc[0], edge.y_val.iloc[0], 0, theta)[2]
#     rotation_left = rotate_points(450-edge.left_peak.iloc[0], edge.y_val.iloc[0], 0, theta)
#     rotation_left = np.array(rotation_left)
#     # rotation_right = rotate_points(450-edge.right_peak.iloc[0], edge.y_val.iloc[0], 0, theta)
#     return rotation_left
    
    
# # edge_results.groupby(['y_val', 'frame']).apply(lambda x: rotate_img(angle_per_frame, x.frame, x))

# threeD_render = pd.DataFrame()
# threeD_render['left_edge'] = edge_results.groupby(['y_val', 'frame']).apply(lambda x: rotate_img(angle_per_frame, x.frame, x))
# # threeD_render['left_edge'] = edge_results.groupby(['y_val']).apply(lambda x: rotate_img(angle_per_frame, x.frame, x[x.frame==0]))

# new_names = ['x','y','z']
# threeD_render[new_names] = threeD_render.apply(lambda x: x.left_edge, axis=1, result_type='expand')



# #%%
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # ax.scatter3D(threeD_render.x,threeD_render.z,threeD_render.y, cmap='Greens')


# df = threeD_render
# # ax.scatter3D(df[(df.y>85)&(df.y<220)].x, df[(df.y>85)&(df.y<220)].y, df[(df.y>85)&(df.y<220)].z)
# ax.plot_trisurf(df.x,df.y,df.z)
# ax.axis(xmin=-400,xmax=400, ymin = 0, ymax = 400)
# # ax.y_lim(-400,400)
# # ax.scatter3D(df.x, df.y, df.z, cmap='jet', linewidth=0.2)

# # threeD_render['x'] = threeD_render.left_edge
# # test =edge_results.groupby(['y_val', 'frame']).apply(lambda x: rotate_img(angle_per_frame, x.frame, x))
# # for i in test[0]:
# #     plt.plot(i[0][0], i[0][2], 'ro')
# #     plt.plot(i[1][0], i[1][2])


# #%%

# ###Connect points on a plane and make a surface ###

# from mpl_toolkits import mplot3d


# # threeD_render
# for i in threeD_render.left_edge:
#     ax.scatter3D(i[0],i[2],i[1], cmap='Greens')






# #%%
# ### Left to right approach ###
# for y in edge_results.y_val.unique():
#     left_peak = edge_results[edge_results.y_val==y].left_peak.iloc[0]
#     right_peak = edge_results[edge_results.y_val==y].right_peak.iloc[0]
    
#     #Crop out any contributions from the lens#
#     if left_peak < 258:
#         continue
#     if right_peak > 660:
#         continue
    
#     D = right_peak-left_peak
#     vol += np.pi*(D/2)**2
    

# #Bottom to Top approach###
# # for peak in edge_results.peak.unique():

# #     if len(edge_results[edge_results.peak==peak])== 1:
# #         continue
# #     else:
# #         x = edge_results[edge_results.peak==peak].x_val
# #         plt.plot(x.min(), peak, 'bo')
# #         plt.plot(x.max(), peak,'bo')
    
# #         D = x.max()-x.min()
# #         vol += np.pi*(D/2)**2

# # print(vol)
# #%%
# ### Create points on a plane ###


