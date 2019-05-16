# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:17:33 2019

@author: surya
"""

from __future__ import print_function

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


'''

fig, (ax1, ax2) = plt.subplots(2, 1)

img = nib.load(
    "D:\\IndividualProject\\Hammersmith\\mls_label_brain\\3t250.nii.gz")

X = img.get_data()

tracker = IndexTracker(ax1, X[:,:,:,0])
#tracker = IndexTracker(ax1, X)

img2 = nib.load(
    "D:\\IndividualProject\\Hammersmith\\mls_label_brain\\3t633.nii.gz")

Y = img2.get_data()
tracker2 = IndexTracker(ax2, Y[:, :, :, 0])
#tracker2 = IndexTracker(ax2, Y)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
fig.canvas.mpl_connect('scroll_event', tracker2.onscroll)
plt.show()
#'''
