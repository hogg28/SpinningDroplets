# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:45:34 2022

@author: johna
"""

import numpy as np
import pandas as pd

edges = pd.read_pickle(directory)

rotation_speed = 1674/60
frame_rate = 27.85
angle_per_frame = rotation_speed*1/frame_rate*360

# angle_per_frame = 7.2

import math

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotate_points(x, y, z, theta, centre):
    axis = [0,centre,0]
    v = [x, y, z]
    
    return np.dot(rotation_matrix(axis, theta), v)

def rotate_img(angle, frame, edge, centre):
    theta = angle*frame
    # z = rotate_points(edge.left_peak.iloc[0], edge.y_val.iloc[0], 0, theta)[2]
    rotation_left = rotate_points(centre-edge.left_peak.iloc[0], edge.y_val.iloc[0], 0, theta, centre)
    rotation_left = np.array(rotation_left)
    # rotation_right = rotate_points(450-edge.right_peak.iloc[0], edge.y_val.iloc[0], 0, theta)
    return rotation_left
    
    
# edge_results.groupby(['y_val', 'frame']).apply(lambda x: rotate_img(angle_per_frame, x.frame, x))

threeD_render = pd.DataFrame()
threeD_render['left_edge'] = edge_results.groupby(['y_val', 'frame']).apply(lambda x: rotate_img(angle_per_frame, x.frame, x, centre))
# threeD_render['left_edge'] = edge_results.groupby(['y_val']).apply(lambda x: rotate_img(angle_per_frame, x.frame, x[x.frame==0]))

new_names = ['x','y','z']
threeD_render[new_names] = threeD_render.apply(lambda x: x.left_edge, axis=1, result_type='expand')



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