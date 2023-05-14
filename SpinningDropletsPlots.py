# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:16:42 2022

@author: KDV Lab
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
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

# sns.palplot(sns.color_palette("muted"))


data_file_1 = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/3SDS/19012023/30minAccumulation_3/Data_9.csv'
data_file_2 = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/3SDS/19012023/30minAccumulation_2/Data_9.csv'
data_file_3 = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/3SDS/19012023/15minAccumulation/Data_9.csv'
data_file_11 = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/3SDS/14022023/30minAccumulation_Large/Data_9.csv'

data_file_4 = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/1SDS/24012023/30minAccumulation_Large/Data_9.csv'
data_file_5 = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/1SDS/24012023/15minAccumulation_Large/Data_9.csv'

data_file_6 = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/2SDS/08022023/30minAccumulation_large/Data_9.csv'
data_file_7 = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/2SDS/08022023/15minAccumulation_large/Data_9.csv'
data_file_8 = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/2SDS/09022023/15minAcummulationLarge/Data_9.csv'
data_file_9 = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/2SDS/09022023/10minAccumulation_Small/Data_9.csv'
data_file_10 = 'C:/Users/johna/OneDrive/Documents/GradSchool/KDVLab/SpinningDrops/2SDS/09022023/15minAccumulationLarge_10minAccumulationSmall/Data_9.csv'

data_1 = pd.read_csv(data_file_1, index_col=0)
data_2 = pd.read_csv(data_file_2, index_col=0)
data_3 = pd.read_csv(data_file_3, index_col=0)
data_4 = pd.read_csv(data_file_4, index_col=0)
data_5 = pd.read_csv(data_file_5, index_col=0)
data_6 = pd.read_csv(data_file_6, index_col=0)
data_7 = pd.read_csv(data_file_7, index_col=0)
data_8 = pd.read_csv(data_file_8, index_col=0)
data_9 = pd.read_csv(data_file_9, index_col=0)
data_10 = pd.read_csv(data_file_10, index_col=0)
data_11 = pd.read_csv(data_file_11, index_col=0)

data_list = ((data_1,data_2, data_3, data_4, data_5, data_6, data_7,data_8, data_9, data_10, data_11))
# data_list = ((data_1,data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9))

for data in data_list:
    V_f = data[(data.ramp == 'down') & (data.rotation_speed > 18) & (data.rotation_speed <31)].vol.mean()
    S_f = data[(data.ramp == 'down') & (data.rotation_speed > 34) ].SA.mean()
    Size_f = data[(data.ramp == 'down') & (data.rotation_speed > 31) ]['size'].mean()
    if data.SDS.iloc[0] == 1:
        data['delta'] = 23.79
        data['Cm'] = 27
    elif data.SDS.iloc[0] == 2:
        data['delta'] = 36.08
        data['Cm'] = 63
    elif data.SDS.iloc[0] ==3:
        data['delta'] = 45.14
        data['Cm'] = 99
    data['delta^2'] = data['delta']**2
    data['rotation_speed^2'] = data['rotation_speed']**2
    data['delta_omega'] = data['delta']*(data['rotation_speed']**2)
    data['omega_g'] = data['rotation_speed^2']/9.8
    data['norm_vol'] = data['vol']/V_f
    data['norm_SA'] = data['SA']/S_f
    data['norm_vol_std'] = data['vol_std']/V_f
    data['norm_size'] = data['size']/Size_f
    data['aspect_ratio'] = data['height']/(data['width'])
    # data['height'] = data['height']*2.5
    
    data['aspect_ratio_std'] = data['aspect_ratio']*np.sqrt((data['height_std']/data['height'])**2+(data['width_std']/data['width'])**2)
    data['theory'] = (1/(((data['width']*2.5)**2)*(data['height']*2.5) ))+((2*data['delta^2'])/(3*((data['width']*2.5)**3)*(data['height']*2.5)))
    # data['aspect_ratio'] = data['width']/data['height']
    data['shape'] = data['width']**2
    # H_f = data[data.rotation_speed>38].height.mean()/(data[data.rotation_speed>38].vol**(1/3)).mean()*data['delta']
    H_f = data[data.rotation_speed>38].vol.mean()**(1/3)*data['delta']
    data['vol'] = data['vol']*(2.5**3)
    data['vol_std'] = data['vol_std']*(2.5**3)
    
    # H_f = data[data.rotation_speed>38].height.mean()
    # H_i = data[(data.rotation_speed<13.5) & (data.ramp == 'up')].aspect_ratio.mean()
    # H_i = data[(data.rotation_speed<13.5) & (data.ramp == 'up')]['height'].mean()
    # data['norm_aspect_ratio'] = data['height']*(data['width']**2)
    # H_f = data[data.rotation_speed>38].norm_aspect_ratio.mean()
    # data['norm_aspect_ratio'] = data['aspect_ratio']/H_f
    # data['norm_aspect_ratio'] = data['aspect_ratio']/(data['vol']**(1/3))
    # data['norm_aspect_ratio'] = data['aspect_ratio']/H_f
    # data['norm_aspect_ratio'] = data['aspect_ratio']*data['radius']/data['delta']
    # data['norm_aspect_ratio'] = data['aspect_ratio']/np.sqrt(data['delta'])*(data['vol']**(1/3))
    # data['aspect_ratio_std'] =data['aspect_ratio_std']/H_f/data['radius']*data['delta']/(data['height']*data['width']**2)
    # data['aspect_ratio'] = data['aspect_ratio']*(data['height']*data['width']**2)
    
    
    # data['aspect_ratio_std'] =data['aspect_ratio_std']/H_f#/(data['vol']**(1/3))
    # V_f = (data[(data.ramp == 'down')& (data.rotation_speed >30)]['vol'].mean()**(1/3))
    V_f = (data[(data.ramp == 'down')& (data.rotation_speed >30)]['vol'].mean())
    # H_f = data[(data.ramp == 'down')& (data.rotation_speed >30)]['height'].mean()
    # W_f = data[(data.ramp == 'down')& (data.rotation_speed >30)]['height'].mean()
    # V = np.pi/2 * data['height']*(data['width']/2)**2
    
    
    # V_f = np.pi/2 * data[(data.ramp == 'down')& (data.rotation_speed >30)]['height'].mean()*(data['width'][(data.ramp == 'down')& (data.rotation_speed >30)].mean()/2)**2
    # data['aspect_ratio'] = data['aspect_ratio']/H_f
    # data['aspect_ratio'] = data['height']
    # data['aspect_ratio'] = data['height']*data['width']**2 / data['vol']
    # data['aspect_ratio'] = (data['vol']**(1/3)-(data[data.rotation_speed>35].vol.mean()**(1/3)))/data['delta']
    # data['aspect_ratio'] = ((data['height']/(data['vol']**(1/3))) / (data[data.rotation_speed>38]['height'].mean()/(data[data.rotation_speed>38]['vol'].mean()**(1/3))))
    # data['aspect_ratio'] = (((data['vol']**(1/3))) / ((data[data.rotation_speed>38]['vol'].mean()**(1/3))))
    # data['aspect_ratio'] = data['height']/data['width'] / (data[data.rotation_speed>38]['vol'].mean()**(1/3))
    # data['aspect_ratio'] = data['height']
    # data['aspect_ratio'] = (data['vol']**(1/3)) (data[(data.ramp == 'down')& (data.rotation_speed >30)]['vol'].mean()**(1/3))
    
    
    data['aspect_ratio'] = (V_f*(2.5**3)-(data['vol']*(2.5**3))**(1/3))
    data['aspect_ratio'] = (V_f-(data['vol'])**(1/3))
    
    
    data['aspect_ratio'] = (V_f-data['vol']**(1/3))/data['delta']
    # data['aspect_ratio'] = data['height']/data['width'] - H_f/W_f
    
    # data['aspect_ratio'] = data['norm_aspect_ratio']/data['delta']*data['radius']
    data['vol_std'] = data['vol_std']/np.sqrt(199)
    data['aspect_ratio_std'] = (1/3)*data['vol']**(-2/3)*(data['vol_std'])/data['delta']
    # data['aspect_ratio_std'] = data['aspect_ratio_std']/data['delta']
    
    # data['aspect_ratio'] = data['height']/(data['width'])
    # data['aspect_ratio_std'] = data['aspect_ratio']*np.sqrt((data['height_std']/data['height'])**2+(data['width_std']/data['width'])**2)
    
    data['aspect_ratio'] = data['vol']/V_f
    data['aspect_ratio_std'] = data['vol_std']*(2.5**3)/np.sqrt(199)/V_f
    # data['aspect_ratio_std'] = data['aspect_ratio']*np.sqrt(((data['vol_std']*(2.5**3)/np.sqrt(199))/data['vol'])**2+(data['width_std']/data['width'])**2)
    
    
    
    

data = pd.concat((data_list))
# data = pd.concat((data_1,data_2, data_3, data_4, data_5, data_6, data_7, data_10))
# data = pd.concat((data_1,data_2, data_3, data_4, data_5, data_6, data_7, data_8))


data['force'] = data['rotation_speed']**2-data['SDS']-data['vol']
data['rad_curve'] = (data['width']**2+data['height']**2)/(2*data['height'])
data['SA_std'] = data['SA_std'] /np.sqrt(199)
data['vol_std'] = data['vol_std'] /np.sqrt(199)
data['SA/V'] = data['SA']/data['vol']
data['SA/V_std'] = data['SA/V']*np.sqrt((data['SA_std']/data['SA'])**2+(data['vol_std']/data['vol'])**2)/np.sqrt(199)
data['SA/V_Norm'] = data['norm_SA']/data['norm_vol']
data['rotation_speed_norm'] = data['rotation_speed']/data['SDS']

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Get Unique continents
color_labels = data['loop'].unique()

# List of colors in the color palettes
rgb_values = sns.color_palette("Set2", 6)

# Map continents to the colors
color_map = dict(zip(color_labels, rgb_values))
# color_map = {'a': (0.9882352941176471, 0.5529411764705883, 0.3843137254901961)}

# Finally use the mapped values

up = data[data.ramp =='up']
down = data[data.ramp == 'down']

p = sns.light_palette("seagreen")
p1 = sns.color_palette("ch:s=.25,rot=-.25")
p2 = sns.cubehelix_palette(8)

#%%
fig, ax = plt.subplots()

# data_6=data_2

plt_data = data_6[(data_6.ramp=='up')& ((data_6.rotation_speed <13.5))]
# plt_data = data_6[(data_6.ramp=='up')& ((data_6.rotation_speed <13.5) | (data_6.rotation_speed ==25.5) )]
# plt_data = data_6[(data_6.ramp=='up')& ((data_6.rotation_speed <13.5) | (data_6.rotation_speed ==25.5) | (data_6.rotation_speed >38))]
# up_data = data_6[(data_6.ramp=='up')]
# plt_data = data_6[(data_6.ramp=='down')& ((data_6.rotation_speed >38))]
# plt_data = data_6[(data_6.ramp=='down')& ((data_6.rotation_speed ==25.5)| (data_6.rotation_speed >38))]
# plt_data = data_6[(data_6.ramp=='down')& ((data_6.rotation_speed <13.5) | (data_6.rotation_speed ==25.5) | (data_6.rotation_speed >38))]




# plt_data = pd.concat((up_data,plt_data))
plt_data = data_6





sns.scatterplot(data = plt_data,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p2[1],
                markers = ['^','v'],
                legend =False,
                s = 300)

plt.errorbar(plt_data.rotation_speed, plt_data.aspect_ratio, yerr = plt_data.aspect_ratio_std, xerr = 1, color = p2[1],ls='',capsize = 4)


# sns.scatterplot(data = data_6,
#                 x = "rotation_speed",
#                 y = "norm_aspect_ratio",
#                 style = "ramp",
#                 # hue = "Trial",
#                 # palette= p2,
#                 color = p1[2],
#                 markers = ['^', 'v'],
#                 # legend =True,
#                 s = 500)
# ax.set_ylabel(r'$h/w$', fontsize = 24, labelpad = 1)
ax.set_ylabel(r'$V$ ($\mu$m$^3)$', fontsize = 24, labelpad = 1)
ax.set_xlabel(r'$\omega$ (Hz)', fontsize = 24, labelpad = -3)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.set_ylim([0.14,0.39])
ax.set_xlim([12,41])
# plt.rc('xtick',labelsize=11)
# plt.rc('ytick',labelsize=11)
fig.tight_layout(pad = 2)

ax.tick_params(axis='x', labelsize=18, length = 4, pad=1)
ax.tick_params(axis='y', labelsize=18, length = 4, pad =1)

plt.show()

#%%

'''
3% SDS
'''

fig, ax = plt.subplots()

# data_1 = data_1[data_1.ramp =='up']
# data_2 = data_2[data_2.ramp =='up']
# data_3 = data_3[data_3.ramp =='up']
# data_11 = data_11[data_11.ramp =='up']

# sns.scatterplot(data = data_1,
#                 x = "rotation_speed",
#                 y = "aspect_ratio",
#                 style = "ramp",
#                 # hue = "Trial",
#                 # palette= p2,
#                 color = p2[0],
#                 markers = ['^','v'],
#                 legend =False,
#                 s = 300)
sns.scatterplot(data = data_2,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p2[1],
                markers = ['^','v'],
                legend =False,
                s = 300)
# sns.scatterplot(data = data_3,
#                 x = "rotation_speed",
#                 y = "aspect_ratio",
#                 style = "ramp",
#                 # hue = "Trial",
#                 # palette= p2,
#                 color = p2[3],
#                 markers = ['^','v'],
#                 legend =False,
#                 s = 300)
# sns.scatterplot(data = data_11,
#                 x = "rotation_speed",
#                 y = "aspect_ratio",
#                 style = "ramp",
#                 # hue = "Trial",
#                 # palette= p2,
#                 color = p2[2],
#                 markers = ['^','v'],
#                 legend =False,
#                 s = 300)



# plt.errorbar(data_1.rotation_speed, data_1.aspect_ratio, yerr = data_1.aspect_ratio_std, xerr = 1, color = p2[0],ls='',capsize = 4)
# plt.errorbar(data_2.rotation_speed, data_2.aspect_ratio, yerr = data_2.aspect_ratio_std, xerr = 1, color = p2[1],ls='',capsize = 4)
# plt.errorbar(data_3.rotation_speed, data_3.aspect_ratio, yerr = data_3.aspect_ratio_std, xerr = 1, color = p2[3],ls='',capsize = 4)
plt.errorbar(data_11.rotation_speed, data_11.aspect_ratio, yerr = data_11.aspect_ratio_std, xerr = 1, color = p2[2],ls='',capsize = 4)



# ax.set_ylim([0.1,0.49])
ax.set_xlim([12,41])
# ax.set_ylim([845,930])

ax.set_ylabel(r'$h/w$', fontsize = 24, labelpad = 1)
ax.set_ylabel(r'$\frac{\left(V_f^{\frac{1}{3}} - V^{\frac{1}{3}}\right)}{\delta}$', fontsize = 16, labelpad = 1)
ax.set_ylabel(r'$V$ ($\mu$m$^3)$', fontsize = 24, labelpad = 1)
# ax.set_ylabel(r'$V_f^{\frac{1}{3}} - V^{\frac{1}{3}}$', fontsize = 16, labelpad = 1)
# ax.set_ylabel(r'$V^{1/3}$', fontsize = 24, labelpad = 1)

ax.set_xlabel(r'$\omega \mathrm{[Hz]}$', fontsize = 16, labelpad = -3)
ax.set_ylabel(r'$V$ ($\mu$m$^3)$', fontsize = 24, labelpad = 1)
ax.set_ylabel(r'$V/V_f$', fontsize = 24, labelpad = 1)
ax.set_xlabel(r'$\omega$ [Hz]', fontsize = 24, labelpad = -3)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.set_ylim([-70,15])
ax.set_xlim([12,41])
ax.set_ylim([0.92,1.25])
# plt.rc('xtick',labelsize=11)
# plt.rc('ytick',labelsize=11)
fig.tight_layout(pad = 2)

ax.tick_params(axis='x', labelsize=18, length = 4, pad=1)
ax.tick_params(axis='y', labelsize=18, length = 4, pad =1)

plt.show()
#%%
'''
2%SDS Data
'''

fig, ax = plt.subplots()

# sns.scatterplot(data = data_6,
#                 x = "rotation_speed",
#                 y = "aspect_ratio",
#                 style = "ramp",
#                 # hue = "Trial",
#                 # palette= p2,
#                 color = p1[0],
#                 markers = ['^','v'],
#                 legend =False,
#                 s = 300)

# sns.scatterplot(data = data_7,
#                 x = "rotation_speed",
#                 y = "aspect_ratio",
#                 style = "ramp",
#                 # hue = "Trial",
#                 # palette= p2,
#                 color = p1[1],
#                 markers = ['^','v'],
#                 legend =False,
#                 s = 300)

# sns.scatterplot(data = data_8,
#                 x = "rotation_speed",
#                 y = "aspect_ratio",
#                 style = "ramp",
#                 # hue = "Trial",
#                 # palette= p2,
#                 color = p1[2],
#                 markers = ['^','v'],
#                 legend =False,
#                 s = 300)

# sns.scatterplot(data = data_9,
#                 x = "rotation_speed",
#                 y = "aspect_ratio",
#                 style = "ramp",
#                 # hue = "Trial",
#                 # palette= p2,
#                 color = p1[3],
#                 markers = ['^','v'],
#                 legend =False,
#                 s = 300)

sns.scatterplot(data = data_10,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p1[4],
                markers = ['^','v'],
                legend =False,
                s = 300)

# plt.errorbar(data_6.rotation_speed, data_6.aspect_ratio, yerr = data_6.aspect_ratio_std, xerr = 1, color = p1[0],ls='',capsize = 4)
plt.errorbar(data_7.rotation_speed, data_7.aspect_ratio, yerr = data_7.aspect_ratio_std, xerr = 1, color = p1[1],ls='',capsize = 4)
# plt.errorbar(data_8.rotation_speed, data_8.aspect_ratio, yerr = data_8.aspect_ratio_std, xerr = 1, color = p1[2],ls='',capsize = 4)
# plt.errorbar(data_9.rotation_speed, data_9.aspect_ratio, yerr = data_9.aspect_ratio_std, xerr = 1, color = p1[3],ls='',capsize = 4)
# plt.errorbar(data_10.rotation_speed, data_10.aspect_ratio, yerr = data_10.aspect_ratio_std, xerr = 1, color = p1[4],ls='',capsize = 4)



# ax.set_ylim([0.14,0.31])
# ax.set_ylim([0.37,1.2])
ax.set_xlim([12,41])
# ax.set_ylim([0.8,2.7])


ax.set_ylabel(r'$h/w$', fontsize = 24, labelpad = 1)
ax.set_xlabel(r'$\omega$ [Hz]', fontsize = 24, labelpad = -3)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_ylabel(r'$\frac{\left(V_f^{\frac{1}{3}} - V^{\frac{1}{3}}\right)}{\delta}$', fontsize = 16, labelpad = 1)
# ax.set_ylabel(r'$V^{1/3}$', fontsize = 24, labelpad = 1)

ax.set_xlabel(r'$\omega \mathrm{[Hz]}$', fontsize = 16, labelpad = -3)
ax.set_ylabel(r'$V/V_f$', fontsize = 24, labelpad = 1)
ax.set_xlabel(r'$\omega$ [Hz]', fontsize = 24, labelpad = -3)
# ax.set_ylim([0.14,0.31])
# ax.set_ylim([-70,15])
# ax.set_ylim([-0.8,0.15])
ax.set_xlim([12,41])
ax.set_ylim([0.92,1.25])
# ax.set_ylim([-1.5,0.4])
# plt.rc('xtick',labelsize=11)
# plt.rc('ytick',labelsize=11)
fig.tight_layout(pad = 2)

ax.tick_params(axis='x', labelsize=18, length = 4, pad=1)
ax.tick_params(axis='y', labelsize=18, length = 4, pad =1)

plt.show()
#%%
'''
1% SDS Data
'''
fig, ax = plt.subplots()




sns.scatterplot(data = data_5,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p[-3],
                markers = ['^','v'],
                legend =False,
                s = 300)
sns.scatterplot(data = data_4,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p[-1],
                markers = ['^','v'],
                legend =False,
                s = 300)

plt.errorbar(data_5.rotation_speed, data_5.aspect_ratio, yerr = data_5.aspect_ratio_std, xerr = 1, color = p[-3],ls='',capsize = 4)
plt.errorbar(data_4.rotation_speed, data_4.aspect_ratio, yerr = data_4.aspect_ratio_std, xerr = 1, color = p[-1],ls='',capsize = 4)

# 


# ax.set_ylim([0.14,0.31])
ax.set_xlim([12,41])
# ax.set_ylim([0.37,1.2])
# ax.set_ylim([0.8,2.7])
ax.set_xlim([12,41])

ax.set_ylabel(r'$h/w$', fontsize = 24, labelpad = 1)
ax.set_xlabel(r'$\omega$ [Hz]', fontsize = 24, labelpad = -3)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_ylabel(r'$\frac{\left(V_f^{\frac{1}{3}} - V^{\frac{1}{3}}\right)}{\delta}$', fontsize = 16, labelpad = 1)
# ax.set_ylabel(r'$V^{1/3}$', fontsize = 24, labelpad = 1)

ax.set_xlabel(r'$\omega \mathrm{[Hz]}$', fontsize = 16, labelpad = -3)
# ax.set_ylim([0.14,0.31])
# ax.set_ylim([-70,15])
# ax.set_ylim([-0.8,0.15])
# ax.set_ylim([0.14,0.31])
ax.set_xlim([12,41])
ax.set_ylim([0.92,1.25])
# ax.set_ylim([-1.5,0.4])
# plt.rc('xtick',labelsize=11)
# plt.rc('ytick',labelsize=11)
fig.tight_layout(pad = 2)
ax.set_ylabel(r'$V/V_f$', fontsize = 24, labelpad = 1)
ax.set_xlabel(r'$\omega$ [Hz]', fontsize = 24, labelpad = -3)
ax.tick_params(axis='x', labelsize=18, length = 4, pad=1)
ax.tick_params(axis='y', labelsize=18, length = 4, pad =1)

plt.show()

#%%
fig, ax = plt.subplots()




sns.scatterplot(data = data_1,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p2[0],
                markers = ['^','v'],
                legend =False,
                s = 300)
sns.scatterplot(data = data_2,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p2[1],
                markers = ['^','v'],
                legend =False,
                s = 300)
sns.scatterplot(data = data_3,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p2[2],
                markers = ['^','v'],
                legend =False,
                s = 300)
sns.scatterplot(data = data_11,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p2[3],
                markers = ['^','v'],
                legend =False,
                s = 300)

sns.scatterplot(data = data_6,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p1[0],
                markers = ['^','v'],
                legend =False,
                s = 300)

sns.scatterplot(data = data_7,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p1[1],
                markers = ['^','v'],
                legend =False,
                s = 300)

sns.scatterplot(data = data_8,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p1[2],
                markers = ['^','v'],
                legend =False,
                s = 300)

sns.scatterplot(data = data_9,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p1[3],
                markers = ['^','v'],
                legend =False,
                s = 300)


sns.scatterplot(data = data_4,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p[-1],
                markers = ['^','v'],
                legend =False,
                s = 300)


# sns.scatterplot(data = data_5,
#                 x = "rotation_speed",
#                 y = "aspect_ratio",
#                 style = "ramp",
#                 # hue = "Trial",
#                 # palette= p2,
#                 color = p[1],
#                 markers = ['^','v'],
#                 legend =False,
#                 s = 300)

# data_1 = data_1[data_1.ramp =='up']
# data_2 = data_2[data_2.ramp =='up']
# data_3 = data_3[data_3.ramp =='up']

# plt.errorbar(data_1.rotation_speed, data_1.aspect_ratio, yerr = data_1.aspect_ratio_std, xerr = 1, color = p2[0],ls='',capsize = 4)
# plt.errorbar(data_2.rotation_speed, data_2.aspect_ratio, yerr = data_2.aspect_ratio_std, xerr = 1, color = p2[1],ls='',capsize = 4)
# plt.errorbar(data_3.rotation_speed, data_3.aspect_ratio, yerr = data_3.aspect_ratio_std, xerr = 1, color = p2[2],ls='',capsize = 4)
# plt.errorbar(data_11.rotation_speed, data_11.aspect_ratio, yerr = data_11.aspect_ratio_std, xerr = 1, color = p2[3],ls='',capsize = 4)

# plt.errorbar(data_6.rotation_speed, data_6.aspect_ratio, yerr = data_6.aspect_ratio_std, xerr = 1, color = p1[0],ls='',capsize = 4)
# # plt.errorbar(data_7.rotation_speed, data_7.aspect_ratio, yerr = data_7.aspect_ratio_std, xerr = 1, color = p1[1],ls='',capsize = 4)
# plt.errorbar(data_8.rotation_speed, data_8.aspect_ratio, yerr = data_8.aspect_ratio_std, xerr = 1, color = p1[2],ls='',capsize = 4)
# plt.errorbar(data_9.rotation_speed, data_9.aspect_ratio, yerr = data_9.aspect_ratio_std, xerr = 1, color = p1[3],ls='',capsize = 4)

# plt.errorbar(data_4.rotation_speed, data_4.aspect_ratio, yerr = data_4.aspect_ratio_std, xerr = 1, color = p[-1],ls='',capsize = 4)
# # plt.errorbar(data_5.rotation_speed, data_5.aspect_ratio, yerr = data_5.aspect_ratio_std, xerr = 1, color = p[-2],ls='',capsize = 4)
# # 


# ax.set_ylabel(r'$h/w * \delta/R$')
ax.set_ylabel(r'$(V^{1/3} - V^{1/3}_f)$', fontsize = 24, labelpad = 1)
ax.set_xlabel(r'$\omega$ [Hz]', fontsize = 24, labelpad = -3)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.set_ylim([0.14,0.31])
ax.set_xlim([12,41])
# plt.rc('xtick',labelsize=11)
# plt.rc('ytick',labelsize=11)
fig.tight_layout(pad = 2)
ax.set_ylabel(r'$V/V_f$', fontsize = 24, labelpad = 1)
ax.set_xlabel(r'$\omega$ [Hz]', fontsize = 24, labelpad = -3)
ax.tick_params(axis='x', labelsize=18, length = 4, pad=1)
ax.tick_params(axis='y', labelsize=18, length = 4, pad =1)
# ax.set_ylim([0.14,0.31])
ax.set_xlim([12,41])
# ax.set_ylim([0.1,0.49])
ax.set_xlim([12,41])

plt.show()


#%%
fig, ax = plt.subplots()

data_3per = pd.concat((data_1, data_2, data_3, data_11))
data_3tmp = pd.DataFrame(data_3per.groupby(['ramp','rotation_speed']).aspect_ratio.mean()).reset_index(drop=False)
data_3tmp['aspect_ratio_std'] = pd.DataFrame(data_3per.groupby(['ramp','rotation_speed']).aspect_ratio.std()).reset_index(drop=False)['aspect_ratio']

data_2per = pd.concat((data_6, data_9, data_10))
data_2tmp = pd.DataFrame(data_2per.groupby(['ramp','rotation_speed']).aspect_ratio.mean()).reset_index(drop=False)
data_2tmp['aspect_ratio_std'] = pd.DataFrame(data_2per.groupby(['ramp','rotation_speed']).aspect_ratio.std()).reset_index(drop=False)['aspect_ratio']

data_1per = data[data.SDS==1]
data_1tmp = pd.DataFrame(data_1per.groupby(['ramp','rotation_speed']).aspect_ratio.mean()).reset_index(drop=False)
data_1tmp['aspect_ratio_std'] = pd.DataFrame(data_1per.groupby(['ramp','rotation_speed']).aspect_ratio.std()).reset_index(drop=False)['aspect_ratio']

sns.scatterplot(data = data_3tmp,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p2[3],
                markers = ['^','v'],
                legend =False,
                s = 300)
sns.scatterplot(data = data_2tmp,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p1[3],
                markers = ['^','v'],
                legend =False,
                s = 300)
sns.scatterplot(data = data_1tmp,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p[-3],
                markers = ['^','v'],
                legend =False,
                s = 300)

# data_1 = data_1[data_1.ramp =='up']
# data_2 = data_2[data_2.ramp =='up']
# data_3 = data_3[data_3.ramp =='up']

plt.errorbar(data_3tmp.rotation_speed, data_3tmp.aspect_ratio, yerr = data_3tmp.aspect_ratio_std, xerr = 1, color = p2[3],ls='',capsize = 4)
# plt.errorbar(data_2.rotation_speed, data_2.aspect_ratio, yerr = data_2.aspect_ratio_std, xerr = 1, color = p2[1],ls='',capsize = 4)
# plt.errorbar(data_3.rotation_speed, data_3.aspect_ratio, yerr = data_3.aspect_ratio_std, xerr = 1, color = p2[2],ls='',capsize = 4)
# plt.errorbar(data_11.rotation_speed, data_11.aspect_ratio, yerr = data_11.aspect_ratio_std, xerr = 1, color = p2[3],ls='',capsize = 4)

plt.errorbar(data_2tmp.rotation_speed, data_2tmp.aspect_ratio, yerr = data_2tmp.aspect_ratio_std, xerr = 1, color = p1[3],ls='',capsize = 4)

# plt.errorbar(data_6.rotation_speed, data_6.aspect_ratio, yerr = data_6.aspect_ratio_std, xerr = 1, color = p1[0],ls='',capsize = 4)
# # plt.errorbar(data_7.rotation_speed, data_7.aspect_ratio, yerr = data_7.aspect_ratio_std, xerr = 1, color = p1[1],ls='',capsize = 4)
# plt.errorbar(data_8.rotation_speed, data_8.aspect_ratio, yerr = data_8.aspect_ratio_std, xerr = 1, color = p1[2],ls='',capsize = 4)
# plt.errorbar(data_9.rotation_speed, data_9.aspect_ratio, yerr = data_9.aspect_ratio_std, xerr = 1, color = p1[3],ls='',capsize = 4)

plt.errorbar(data_1tmp.rotation_speed, data_1tmp.aspect_ratio, yerr = data_1tmp.aspect_ratio_std, xerr = 1, color = p[-3],ls='',capsize = 4)

# plt.errorbar(data_4.rotation_speed, data_4.aspect_ratio, yerr = data_4.aspect_ratio.std, xerr = 1, color = p[-1],ls='',capsize = 4)
# # plt.errorbar(data_5.rotation_speed, data_5.aspect_ratio, yerr = data_5.aspect_ratio_std, xerr = 1, color = p[-2],ls='',capsize = 4)
# # 


# ax.set_ylabel(r'$h/w * \delta/R$')
ax.set_ylabel(r'$(V^{1/3} - V^{1/3}_f)$', fontsize = 24, labelpad = 1)
ax.set_xlabel(r'$\omega$ [Hz]', fontsize = 24, labelpad = -3)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_ylabel(r'$\frac{\left(V_f^{\frac{1}{3}} - V^{\frac{1}{3}}\right)}{\delta}$', fontsize = 16, labelpad = 1)

# ax.set_ylabel(r'$V^{1/3}$', fontsize = 24, labelpad = 1)

ax.set_xlabel(r'$\omega \mathrm{[Hz]}$', fontsize = 16, labelpad = -3)
# ax.set_ylim([0.14,0.31])
# ax.set_ylim([-70,15])
# ax.set_ylim([0.14,0.31])
ax.set_xlim([12,41])
ax.set_ylim([-1.5,0.4])
# plt.rc('xtick',labelsize=11)
# plt.rc('ytick',labelsize=11)
fig.tight_layout(pad = 2)

ax.tick_params(axis='x', labelsize=18, length = 4, pad=1)
ax.tick_params(axis='y', labelsize=18, length = 4, pad =1)
# ax.set_ylim([0.14,0.31])
ax.set_xlim([12,41])
# ax.set_ylim([0.1,0.49])
ax.set_xlim([12,41])

plt.show()
#%%
fig, ax = plt.subplots()

'''
Glass Plots

'''

#6,7,8,9,
# data_6['aspect_ratio']



# sns.scatterplot(data = data_9,
#                 x = "rotation_speed",
#                 y = "aspect_ratio",
#                 style = "ramp",
#                 # hue = "Trial",
#                 # palette= p2,
#                 color = p1[3],
#                 markers = ['^','v'],
#                 legend =False,
#                 s = 300)

plt_data = data_10[(data_10.ramp=='up')& ((data_10.rotation_speed <13.5))]
# plt_data = data_10[(data_10.ramp=='up')& ((data_10.rotation_speed <13.5) | (data_10.rotation_speed ==25.5) )]
# plt_data = data_10[(data_10.ramp=='up')& ((data_10.rotation_speed <13.5) | (data_10.rotation_speed ==25.5) | (data_10.rotation_speed >38))]
# up_data = data_10[(data_10.ramp=='up')]
# plt_data = data_10[(data_10.ramp=='down')& ((data_10.rotation_speed >38))]
# plt_data = data_10[(data_10.ramp=='down')& ((data_10.rotation_speed ==25.5)| (data_10.rotation_speed >38))]
# plt_data = data_10[(data_10.ramp=='down')& ((data_10.rotation_speed <13.5) | (data_10.rotation_speed ==25.5) | (data_10.rotation_speed >38))]


# plt_data = pd.concat((up_data,plt_data))
plt_data = data_10


sns.scatterplot(data = plt_data,
                x = "rotation_speed",
                y = "aspect_ratio",
                style = "ramp",
                # hue = "Trial",
                # palette= p2,
                color = p1[5],
                markers = ['^','v'],
                legend =False,
                s = 200)

# sns.scatterplot(data = data_6,
#                 x = "rotation_speed",
#                 y = "aspect_ratio",
#                 style = "ramp",
#                 # hue = "Trial",
#                 # palette= p2,
#                 color = p1[1],
#                 markers = ['^','v'],
#                 legend =False,
#                 s = 200)



# 
# plt.errorbar(data_6.rotation_speed, data_6.aspect_ratio, yerr = data_6.aspect_ratio_std, xerr = 1, color = p1[1],ls='',capsize = 4)

# plt.errorbar(data_7.rotation_speed, data_7.aspect_ratio, yerr = data_7.aspect_ratio_std, xerr = 1, color = p1[2],ls='',capsize = 4)
# 
# plt.errorbar(data_8.rotation_speed, data_8.aspect_ratio, yerr = data_8.aspect_ratio_std, xerr = 1, color = p1[2],ls='',capsize = 4)
# plt.errorbar(data_9.rotation_speed, data_9.aspect_ratio, yerr = data_9.aspect_ratio_std, xerr = 1, color = p1[3],ls='',capsize = 4)
# plt.errorbar(data_10.rotation_speed, data_10.aspect_ratio, yerr = data_10.aspect_ratio_std, xerr = 1, color = p1[5],ls='',capsize = 4)

plt.errorbar(plt_data.rotation_speed, plt_data.aspect_ratio, yerr = plt_data.aspect_ratio_std, xerr = 1, color = p1[5],ls='',capsize = 4)



ax.set_ylim([0.9776155472749845,1.2162222116385766])
ax.set_xlim([12,41])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_ylabel(r'$h/w$', fontsize = 24, labelpad = 1)
ax.set_ylabel(r'$V/V_f$', fontsize = 24, labelpad = 1)
ax.set_xlabel(r'$\omega$ (Hz)', fontsize = 24, labelpad = -3)
# ax.set_xlabel(r'$\omega$ [Hz]', fontsize = 24, labelpad = -3)
# plt.rc('xtick',labelsize=11)
# plt.rc('ytick',labelsize=11)
fig.tight_layout(pad = 2)

ax.tick_params(axis='x', labelsize=18, length = 4, pad=1)
ax.tick_params(axis='y', labelsize=18, length = 4, pad =1)

plt.show()
