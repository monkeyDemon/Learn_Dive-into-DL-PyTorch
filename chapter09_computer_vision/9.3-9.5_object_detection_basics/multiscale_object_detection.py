# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 21:17:13 2020

多尺度目标检测

@author: as
"""
import sys
sys.path.append("../../")

import d2lzh_pytorch as d2l

import torch
import numpy as np
from matplotlib import pyplot as plt


# load catdog image
img = plt.imread('catdog.jpg')
h, w = img.shape[0:2]
print("catdog img shape: h {}, w {}".format(h,w))



def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize((3.5, 2.5))
    fig = plt.imshow(img)

    feature_map_size = (fmap_h, fmap_w)  # (h,w)
    anchors = d2l.MultiBoxPrior(feature_map_size, sizes=s, ratios=[1, 2, 0.5])
    
    bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
    d2l.show_bboxes(fig.axes, anchors * bbox_scale)
    plt.savefig('multiscale_anchor_{}_{}.png'.format(fmap_h, fmap_w))

display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
