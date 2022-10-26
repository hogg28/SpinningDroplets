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
import time
from scipy.optimize import curve_fit
from skimage import feature
import numpy as np
from scipy.ndimage.interpolation import rotate
from skimage.filters import sobel, gaussian, unsharp_mask
import re


rotation_data = pd.read_csv('F:/Johnathan/SpinningDrops/092022/30092022/2SDS/2SDSlarge_170.21Hz/Data.csv')

### Import image sequence ###

mpl.rc('image', cmap='gray')

@pims.pipeline
def preprocess_img(frame):
    
    frame = np.rot90(frame)
    # frame = rotate(frame, angle=6)
    # frame = frame[100:600,50:1020]
    # frame = gaussian(frame, sigma = 3)
    # frame = unsharp_mask(frame, radius = 2, amount = 3)
    # frame *= 255.0/frame.max()
    return frame


directory = 'F:/Johnathan/SpinningDrops/092022/30092022/2SDS/2SDSlarge_170.21Hz/rampup/1554rpm5.5V170.2rampup/'
prefix = '*.tiff'


plt.figure()
frames = preprocess_img(pims.ImageSequence(os.path.join(directory+prefix)))


plt.imshow(frames[20])
#%%
def auto_rotate(frame):
    c=1
    left_side = frame[:,0]
    right_side = frame[:,1024]
    left_slope = []
    right_slope = []
    for i in range(len(left_side)-c):
        left_slope.append((avg_pos[i]-avg_pos[i+c])/(c))
    for i in range(len(right_side)-c):
        right_slope.append((avg_pos[i]-avg_pos[i+c])/(c))

    #find peak on both left and right
    left_peak = signal.find_peaks(-np.array(left_slope), prominence = 2.5)[0]
    right_peak = signal.find_peaks(-np.array(right_slope), prominence = 2.5)[0]
    while np.abs(left_peak - right_peak)>3:
        angle = 1
        left_side = frame[:,0]
        right_side = frame[:,1024]
        left_slope = []
        right_slope = []
        for i in range(len(left_side)-c):
            left_slope.append((avg_pos[i]-avg_pos[i+c])/(c))
        for i in range(len(right_side)-c):
            right_slope.append((avg_pos[i]-avg_pos[i+c])/(c))
        left_peak = signal.find_peaks(-np.array(left_slope), prominence = 2.5)[0]
        right_peak = signal.find_peaks(-np.array(right_slope), prominence = 2.5)[0]
        if np.abs((left_peak - right_peak))<15:
            angle = 0.1
        if left_peak - right_peak > 0:
            frame = rotate(frame, angle=angle)
        elif left_peak - right_peak < 0:
            frame = rotate(frame, angle=-angle)
    return(frame)


#%%

plt.plot(frames[0][:,0])


pos = frames[0][:,0]
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
avg_pos = moving_average(pos, n=20)
slope = []
c=20
for i in range(len(avg_pos)-c):
    slope.append((avg_pos[i]-avg_pos[i+c])/(c))
second_derv = []
for i in range(len(slope)-1):
    second_derv.append((slope[i]-slope[i+1]))
    
right_peak = signal.find_peaks(-np.array(slope), prominence = 2.5)[0]

if right_peak.size == 0:
    right_peak = signal.find_peaks(np.array(second_derv), prominence = 2.5)[0]
# left_peak = signal.find_peaks(np.array(slope), prominence = 2.5)[0]

plt.figure()  
plt.plot(slope)
plt.plot(second_derv)
# plt.plot(right_peak.min())
plt.plot(int(right_peak.max()), slope[int(right_peak.max())], 'ro')
# plt.plot(left_peak.min())
# plt.plot(int(left_peak.min()), slope[int(left_peak.min())], 'ro')

plt.figure()
plt.imshow(frames[0])
plt.plot( 0,right_peak.max()+20, 'ro')
# plt.plot(left_peak.min()+20, 100, 'ro')


#%%
### Detect edges ###


###Bottom to Top###

edge_results = pd.DataFrame(columns=['peak','x_val','frame'])

    
for idx, frame in enumerate(frames[:]):   
    for x in range(0,970):
        pos = frame[:,x]
        c=20
        avg_pos = moving_average(pos, n=c)
        slope = []

        for i in range(len(avg_pos)-c):
            slope.append((avg_pos[i]-avg_pos[i+c])/(c))
        peak = signal.find_peaks(-np.array(slope), prominence = 2.5)[0]

        if right_peak.size == 0:
            peak = np.array([-c])

        edge_results = edge_results.append([{'peak': peak.min() +c,
                                              'x_val': x,
                                              'frame': idx,
                                              },])
        # edge_results = edge_results.concat[peak.max()+20, x, idx]
    print(idx)
        
#%%
plt.figure()
plt.plot(edge_results[edge_results.frame ==25].x_val,edge_results[edge_results.frame ==25].peak)

# plt.plot(edge_results[edge_results.frame ==10].left_peak,edge_results[edge_results.frame ==10].y_val, 'ro')
# plt.plot(edge_results[edge_results.frame ==10].right_peak,edge_results[edge_results.frame ==10].y_val, 'bo')
plt.imshow(frames[25])
# plt.hlines(150, 20, 900, color = 'black', linestyles = 'dashed')


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
test = frames[20]
filled = np.empty((frame.shape[0],frame.shape[1]))
for p in test.peak.unique():
    right = test[test.peak==p].x_val.min()
    left = test[test.peak==p].x_val.max()
    if test[test.peak==p] is empty:
        right = test[test.peak==p-1].x_val.min()
        left = test[test.peak==p-1].x_val.max()
    filled[p, right:left]=1

plt.figure()
plt.imshow(filled)

#%%
# filled = np.empty((frame.shape[0],frame.shape[1]))
# for x in test.x_val.unique():
#     top = test[test.x_val==x].peak.min()
#     bottom = test.peak.min()
#     filled[bottom:top,x]=1

# plt.figure()
# plt.imshow(filled)
#%%


Aggregate_structure = pd.DataFrame()

edges = edge_results.groupby(['frame','peak'])
D = edges.max()-edges.min() 
circ = np.pi*D
vol = np.pi*(D/2)**2

Aggregate_structure['vol'] = vol.groupby('frame').sum()
Aggregate_structure['circ'] = circ.groupby('frame').sum()
Aggregate_structure['frame'] = Aggregate_structure.index

filename = directory.split('/')[-2]
rpm = int(re.search('\d+rpm',filename).group(0).split('rpm')[0])
frame_rate = float(re.search('\d+.\d+ramp',filename).group(0).split('ramp')[0])
rotation_speed = rpm/60

# Aggregate_structure.to_csv(directory + 'shapeData.csv')

data = pd.DataFrame()

data['circ'] = [Aggregate_structure['circ'].mean()]
data['circ_std'] = [Aggregate_structure['circ'].std()]
data['vol'] = [Aggregate_structure['vol'].mean()]
data['vol_std'] = [Aggregate_structure['vol'].std()]
data['rotation_speed'] = [rotation_speed]
data['frame_rate'] = [frame_rate]
data['filename'] = [filename]

rotation_data = rotation_data.append(data)

# rotation_data.to_csv('D:/Johnathan/SpinningDrops/092022/30092022/2SDS/2SDSlarge_170.21Hz/Data.csv')

# Aggregate_structure['vol'] = edge_results.groupby('frame').apply(lambda x: characterize_structure(x))
# Aggregate_structure['frame'] = Aggregate_structure.index

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


