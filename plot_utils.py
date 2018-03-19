#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    collection of utils for plotting

    * Images utils
"""



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
