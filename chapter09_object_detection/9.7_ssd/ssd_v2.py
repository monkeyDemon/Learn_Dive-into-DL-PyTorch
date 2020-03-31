# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:48:36 2020

Single Shot Multibox Detection (SSD)

本代码基于之前介绍的边界框、锚框、多尺度目标检测和数据集等背景知识来构造一个目标检测模型：
单发多框检测（single shot multibox detection，SSD）

@author: as
"""
import os
import sys
sys.path.append("../../")
import d2lzh_pytorch as d2l

import json
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms


# define classification prediction layer
def cls_predictor(input_channels, num_anchors, num_classes):
    return nn.Conv2d(in_channels=input_channels, out_channels=num_anchors * (num_classes + 1), kernel_size=3, padding=1)

# define bounding box prediction layer
def bbox_predictor(input_channels, num_anchors):
    return nn.Conv2d(in_channels=input_channels, out_channels=num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    return block(x)
Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(Y1.shape, Y2.shape)


def flatten_pred(pred):
    # change (B,C,H,W) to (B,H,W,C) and flatten
    return pred.permute(0, 2, 3, 1).reshape(pred.size(0),-1)

def concat_preds(preds):
    return torch.cat(tuple([flatten_pred(p) for p in preds]), dim=1)

# concatenating predictions for Multiple Scales
concat_Y = concat_preds([Y1, Y2])
print(concat_Y.shape)


# Height and Width Downsample Block
def down_sample_blk(input_channels, num_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(num_features=num_channels))
        blk.append(nn.ReLU())
        input_channels=num_channels
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    blk = nn.Sequential(*blk)
    return blk

temp_Y = forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10))
print(temp_Y.shape)


# Base Network Block
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    blk = nn.Sequential(*blk)
    return blk

temp_Y = forward(torch.zeros((2, 3, 256, 256)), base_net())
print(temp_Y.shape) 


# define feature blocks
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk


def blk_forward(X, blk, sizes, ratios, cls_predictor, bbox_predictor):
    Y = blk(X)

    feature_map_size = (Y.size(2), Y.size(3))  # (h,w)
    anchors = d2l.MultiBoxPrior(feature_map_size, sizes=sizes, ratios=ratios)

    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)



sizes = [[0.2*256, 0.272*256], [0.37*256, 0.447*256], [0.54*256, 0.619*256],
         [0.71*256, 0.79*256], [0.88*256, 0.961*256]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


# define the complete model: TinySSD
class TinySSD(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(TinySSD, self).__init__()
        
        input_channels_cls = 128
        input_channels_bbox = 128
        self.num_classes = num_classes
        
        self.blk_0 = get_blk(0)    # backbone, output: (*,64,32,32)
        self.blk_1 = get_blk(1)    # downsample, output: (*,128,16,16)
        self.blk_2 = get_blk(2)    # downsample, output: (*,128,8,8)
        self.blk_3 = get_blk(3)    # downsample, output: (*,128,4,4)
        self.blk_4 = get_blk(4)    # global pooling, output: (*,128,1,1)
        
        self.cls_0 = cls_predictor(64, num_anchors, num_classes)
        self.cls_1 = cls_predictor(input_channels_cls, num_anchors, num_classes)
        self.cls_2 = cls_predictor(input_channels_cls, num_anchors, num_classes)
        self.cls_3 = cls_predictor(input_channels_cls, num_anchors, num_classes)
        self.cls_4 = cls_predictor(input_channels_cls, num_anchors, num_classes)
        
        self.bbox_0 = bbox_predictor(64, num_anchors)
        self.bbox_1 = bbox_predictor(input_channels_bbox, num_anchors)
        self.bbox_2 = bbox_predictor(input_channels_bbox, num_anchors)
        self.bbox_3 = bbox_predictor(input_channels_bbox, num_anchors)
        self.bbox_4 = bbox_predictor(input_channels_bbox, num_anchors)
    
    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        
        X, anchors[0], cls_preds[0], bbox_preds[0] = blk_forward(X, self.blk_0, sizes[0], ratios[0],
                                                                    self.cls_0, self.bbox_0)
        
        X, anchors[1], cls_preds[1], bbox_preds[1] = blk_forward(X, self.blk_1, sizes[1], ratios[1],
                                                                    self.cls_1, self.bbox_1)
            
        X, anchors[2], cls_preds[2], bbox_preds[2] = blk_forward(X, self.blk_2, sizes[2], ratios[2],
                                                                    self.cls_2, self.bbox_2)    
        
        X, anchors[3], cls_preds[3], bbox_preds[3] = blk_forward(X, self.blk_3, sizes[3], ratios[3],
                                                                    self.cls_3, self.bbox_3)    
        
        X, anchors[4], cls_preds[4], bbox_preds[4] = blk_forward(X, self.blk_4, sizes[4], ratios[4],
                                                                    self.cls_4, self.bbox_4)    
        total_anchors_num = 5444  # (32^2 + 16^2 + 8^2 + 4^2 + 1) * num_anchors 
        return (torch.cat(anchors, dim=0), concat_preds(cls_preds).reshape((-1, total_anchors_num, self.num_classes + 1)), 
                concat_preds(bbox_preds))


# now we can create a SSD model
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
 
net = TinySSD(3, num_classes=1)
net.apply(init_weights)

X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)


# --------------training-----------------

# Data Reading and Initialization
# d2l.download_and_preprocess_data()

batch_size = 32
data_dir = '../../dataset/pikachu/'
train_dataset = d2l.PIKACHU(data_dir, 'train')
val_dataset = d2l.PIKACHU(data_dir, 'val')

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=4)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=4)


# TODO: choose gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# init model 
net = TinySSD(3, num_classes=1)
net.apply(init_weights)
net = net.to(device)

learning_rate = 1e-3
weight_decay = 5e-4
optimizer = optim.SGD(net.parameters(), lr = learning_rate, weight_decay=weight_decay)

# Define Loss and Evaluation Functions
id_cat = dict()
id_cat[0] = 'pikachu'

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, device="cuda:0", eps=1e-10):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.eps = eps

    def forward(self, input, target):
        p = torch.sigmoid(input)
        pt = p * target.float() + (1.0 - p) * (1 - target).float()
        alpha_t = (1.0 - self.alpha) * target.float() + self.alpha * (1 - target).float()
        loss = - 1.0 * torch.pow((1 - pt), self.gamma) * torch.log(pt + self.eps)
        return loss.sum()
    
class SSDLoss(nn.Module):
    def __init__(self, loc_factor, jaccard_overlap, device = "cuda:0", **kwargs):
        super().__init__()
        self.fl = FocalLoss(**kwargs)
        self.loc_factor = loc_factor
        self.jaccard_overlap = jaccard_overlap
        

        
        self.device = device

    def one_hot_encoding(labels, num_classes):
        return torch.eye(num_classes)[labels]

    def loc_transformation(x, anchors, overlap_indicies):
        # Doing location transformations according to SSD paper
        return torch.cat([(x[:, 0:1] - anchors[overlap_indicies, 0:1]) / anchors[overlap_indicies, 2:3],
                          (x[:, 1:2] - anchors[overlap_indicies, 1:2]) / anchors[overlap_indicies, 3:4],
                          torch.log((x[:, 2:3] / anchors[overlap_indicies, 2:3])),
                          torch.log((x[:, 3:4] / anchors[overlap_indicies, 3:4]))
                         ], dim=1)

    def forward(self, class_hat, bb_hat, class_true, bb_true, anchors):        
        loc_loss = 0.0
        class_loss = 0.0
    
        
        for i in range(len(class_true)):  # Batch level

            class_hat_i = class_hat[i, :, :]

            bb_true_i = bb_true[i].float()
            
            class_true_i = class_true[i]            
            
            class_target = torch.zeros(class_hat_i.shape[0]).long().to(self.device)
            
            overlap_list = d2l.find_overlap(bb_true_i.squeeze(0), anchors, self.jaccard_overlap)
            
            temp_loc_loss = 0.0
            for j in range(len(overlap_list)):  # BB level
                overlap = overlap_list[j]
                class_target[overlap] = class_true_i[0, j]

                input_ = bb_hat[i, overlap, :]
                target_ = SSDLoss.loc_transformation(bb_true_i[0, j, :].expand((len(overlap), 4)), anchors, overlap)

                temp_loc_loss += F.smooth_l1_loss(input=input_, target=target_, reduction="sum") / len(overlap)
            loc_loss += temp_loc_loss / class_true_i.shape[1]

            class_target = SSDLoss.one_hot_encoding(class_target, len(id_cat) + 1).float().to(self.device)
            class_loss += self.fl(class_hat_i, class_target) / class_true_i.shape[1]

        loc_loss = loc_loss / len(class_true)
        class_loss = class_loss / len(class_true)
        loss = class_loss + loc_loss * self.loc_factor

        return loss, loc_loss, class_loss
