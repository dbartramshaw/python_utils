#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    collection of utils for plotting

    * Images utils
"""


##############################
# Plot Visualisation
##############################

# Title positioning
plt.title('Title', fontsize=14, y=1.05)


# 3d plotting
%matplotlib inline
import mpld3
mpld3.enable_notebook()



##############################
# Image plotting
##############################

# PLot random images frame by frame
import numpy as np
for j in range(0,3):
    img = np.random.normal(size=(100,150))
    plt.figure(1); plt.clf()
    plt.imshow(img)
    plt.title('Number ' + str(j))
    plt.pause(3)



# plot frame by frame
images_loop = test_dict[keys_[2]]
images_loop.shape #(59, 100, 100) 59 images 100x100
for j in images_loop:
    img = np.random.normal(size=(100,150))
    plt.figure(1); plt.clf()
    plt.imshow(j)
    #plt.title('Number ' + str(j))
    plt.pause(0.2)



#Plot points over an image
import matplotlib.pyplot as plt
im = plt.imread(image_name)
implot = plt.imshow(im)

# put a blue dot at (10, 20)
plt.scatter([10], [20])

# put a red dot, size 40, at 2 locations:
plt.scatter(x=[30, 40], y=[50, 60], c='r', s=40)
plt.show()




##############################
# Seaborn Countplot (barplot)
##############################

# barplot
plt.figure(figsize=(15,10))
plt.title('Number of different months bought per customer - 2 year period', fontsize=14)
sns.set_style('whitegrid')
ax = sns.countplot(data=dfs, x = 'total_months')




##############################
# Seaborn Countplot (barplot) - Lablled
##############################
#Total number of months per user
vol_by_totalmonths = dfs.groupby('total_months').size()

#Percentage
per_by_totalmonths =  dfs.groupby('total_months').size() / len(dfs)
per_by_totalmonths = per_by_totalmonths.map(lambda n: '{:,.2%}'.format(n))


plt.figure(figsize=(15,10))
plt.title('Number of different months bought per customer - 2 year period', fontsize=14)
sns.set_style('whitegrid')
ax = sns.countplot(data=dfs, x = 'total_months')

for p, label in zip(ax.patches, per_by_totalmonths):
    ax.annotate(label, (p.get_x()+0, p.get_height()+500))




##############################
# Interactive scatter ATOM
##############################
from matplotlib.pyplot import figure, show
import numpy as np
import matplotlib.pyplot as plt
from plotly import offline as py
import plotly.tools as tls
from numpy.random import rand
py.init_notebook_mode()

if 1: # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)

    x, y, c, s = rand(4, 100)
    def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', ind, npy.take(x, ind), npy.take(y, ind))

    fig = figure()
    ax1 = fig.add_subplot(111)
    col = ax1.scatter(x, y, 100*s, c, picker=True)
    #fig.savefig('pscoll.eps')
    fig.canvas.mpl_connect('pick_event', onpick3)

py.iplot(tls.mpl_to_plotly(plt.gcf()))


##############################
# tooltip show lables hover
##############################
import matplotlib.pyplot as plt
from mpldatacursor import datacursor
import random

fig, ax = plt.subplots()
ax.set_title('Click on a dot to display its label')

# Plot a number of random dots
for i in range(1, 1000):
    ax.scatter([random.random()], [random.random()], label='$ID: {}$'.format(i))

# Use a DataCursor to interactively display the label for a selected line...
datacursor(formatter='{label}'.format)
py.iplot(tls.mpl_to_plotly(plt.gcf()))



############################################
# pairplot - distributions and correlation
############################################
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns
path='/Users/bartramshawd/Documents/datasets/'
white_wine = pd.read_csv(path+'winequality-white.csv', sep=';')
red_wine = pd.read_csv(path+'winequality-red.csv', sep=';')

# store wine type as an attribute
red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'

# bucket wine quality scores into qualitative quality labels
red_wine['quality_label'] = red_wine['quality'].apply(lambda value: 'low'
                                                          if value <= 5 else 'medium'
                                                              if value <= 7 else 'high')
red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'],
                                           categories=['low', 'medium', 'high'])
white_wine['quality_label'] = white_wine['quality'].apply(lambda value: 'low'
                                                              if value <= 5 else 'medium'
                                                                  if value <= 7 else 'high')
white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'],
                                             categories=['low', 'medium', 'high'])

# merge red and white wine datasets
wines = pd.concat([red_wine, white_wine])

# re-shuffle records just to randomize data points
wines = wines.sample(frac=1, random_state=42).reset_index(drop=True)


# Pair-wise Scatter Plots
cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity']
pp = sns.pairplot(wines[cols], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)




'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
%matplotlib inline

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
