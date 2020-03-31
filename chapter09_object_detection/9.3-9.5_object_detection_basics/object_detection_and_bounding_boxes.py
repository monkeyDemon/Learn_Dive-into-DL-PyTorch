# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 21:17:13 2020

目标检测和边界框

@author: as
"""
import sys
sys.path.append("../../")

import d2lzh_pytorch as d2l

import numpy as np
from matplotlib import pyplot as plt


d2l.set_figsize((3.5, 2.5))
img = plt.imread('catdog.jpg')

# bbox is the abbreviation for bounding box
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]

# has save to the d2l package.
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format.
    Convert the bounding box (top-left x, top-left y, bottom-right x, bottom-right y) 
    format to matplotlib format: ((upper-left x, upper-left y), width, height)
    """
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
                        fill=False, edgecolor=color, linewidth=2)


fig = plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
plt.savefig('catdog_with_bbox.png')

