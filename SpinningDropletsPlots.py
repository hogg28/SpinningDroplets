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
import seaborn as sns

# sns.palplot(sns.color_palette("muted"))

data_file = 'F:/Angela/112022/25112022/Data.csv'
date = '25112022'

data = pd.read_csv(data_file, index_col=0)

# Get Unique continents
color_labels = data['loop'].unique()

# List of colors in the color palettes
rgb_values = sns.color_palette("Set2", 6)

# Map continents to the colors
color_map = dict(zip(color_labels, rgb_values))

# Finally use the mapped values

up = data[data.ramp =='up']
down = data[data.ramp == 'down']

# colors = {'initial':'red', 'a':'green', 'b':'blue', 'c':'yellow', 'd':'black', 'e':'orange'}
fig1, ax = plt.subplots()
# ax.scatter(data['rotation_speed'], data['height'], c=data['loop'].map(colors), s = 200)
ax.errorbar(up['rotation_speed'], up['height'], yerr=up['height_std'], ls='', capsize = 1, capthick= 0.25, elinewidth = 0.25, ms=4, mew=0.25, marker = "")
ax.errorbar(down['rotation_speed'], down['height'], yerr=down['height_std'], ls='', capsize = 1, capthick= 0.25, elinewidth = 0.25, ms=4, mew=0.25, marker = "")
ax.scatter(up['rotation_speed'], up['height'], c=up['loop'].map(color_map), s = 75, marker = "^")
ax.scatter(down['rotation_speed'], down['height'], c=down['loop'].map(color_map), s = 75, marker = "v")
leg = plt.legend(['ramp up', 'ramp down'], loc = 0)
ax.add_artist(leg)
# plt.legend(color_map)
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_map.values()]
ax.legend(markers, color_map.keys(), numpoints=1, title = 'Loop', loc = 4)
# ax.legend(data.loop.unique())
# plt.plot(data.rotation_speed, data.height, marker = 'o', ls='')
plt.ylabel('height (px)')
plt.xlabel('rotation_speed (Hz)')
ax.set_title(date+'-height zoomed in')
fig1.tight_layout(pad = 0.2)
plt.show()

fig2, ax = plt.subplots()

ax.errorbar(up['rotation_speed'], up['SA'], yerr=up['SA_std'], ls='', capsize = 1, capthick= 0.25, elinewidth = 0.25, ms=4, mew=0.25, marker = "")
ax.errorbar(down['rotation_speed'], down['SA'], yerr=down['SA_std'], ls='', capsize = 1, capthick= 0.25, elinewidth = 0.25, ms=4, mew=0.25, marker = "")
ax.scatter(up['rotation_speed'], up['SA'], c=up['loop'].map(color_map), s = 75, marker = "^")
ax.scatter(down['rotation_speed'], down['SA'], c=down['loop'].map(color_map), s = 75, marker = "v")
leg = plt.legend(['ramp up', 'ramp down'], loc = 0)
ax.add_artist(leg)
# plt.legend(color_map)
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_map.values()]
ax.legend(markers, color_map.keys(), numpoints=1, title = 'Loop', loc = 4)
# plt.plot(data.rotation_speed, data.SA, marker = 'o', ls='')
plt.ylabel('SA (px^2)')
plt.xlabel('rotation_speed (Hz)')
ax.set_title(date+'-SA')
fig2.tight_layout(pad = 0.2)
plt.show()

fig3, ax = plt.subplots()
ax.errorbar(up['rotation_speed'], up['vol'], yerr=up['vol_std'], ls='', capsize = 1, capthick= 0.25, elinewidth = 0.25, ms=4, mew=0.25, marker = "")
ax.errorbar(down['rotation_speed'], down['vol'], yerr=down['vol_std'], ls='', capsize = 1, capthick= 0.25, elinewidth = 0.25, ms=4, mew=0.25, marker = "")
ax.scatter(up['rotation_speed'], up['vol'], c=up['loop'].map(color_map), s = 75, marker = "^")
ax.scatter(down['rotation_speed'], down['vol'], c=down['loop'].map(color_map), s = 75, marker = "v")
leg = plt.legend(['ramp up', 'ramp down'], loc = 0)
ax.add_artist(leg)
# plt.legend(color_map)
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_map.values()]
plt.legend(markers, color_map.keys(), numpoints=1, title = 'Loop', loc = 2)
# plt.plot(data.rotation_speed, data.vol,  marker = 'o', ls='')
plt.ylabel('volume (px^3)')
plt.xlabel('rotation_speed (Hz)')
ax.set_title(date+'-Volume')
fig3.tight_layout(pad = 0.2)
plt.show()

# plt.figure()
# # data.plot("rotation_speed", "height", color="colour")
# data.plot.scatter(x='rotation_speed',
#                   y='height',
#                   c='colour',
#                   colormap='viridis')
# data.plot(color = 'loop')