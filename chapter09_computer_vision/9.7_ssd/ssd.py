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
from matplotlib import pyplot as plt

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



sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')


# init model 
net = TinySSD(3, num_classes=1)
net.apply(init_weights)
net = net.to(device)

learning_rate = 1e-3
weight_decay = 5e-4
optimizer = optim.SGD(net.parameters(), lr = learning_rate, weight_decay=weight_decay)


# 定义损失函数和评价函数
cls_criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss(input, target) input会自动计算softmax, target会自动进行onehot编码
bbox_criterion = torch.nn.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    """ 计算总体的loss
    cls_preds:    模型预测的每个anchor的类别[batch, anchor_num, class_num + 1]
    cls_labels:   根据groundtruth为每个anchor标注的目标类别(没有进行onehot编码) [batch, anchor_num]
    bbox_preds:   模型预测的对应每个anchor的偏移量 [batch, anchor_num * 4]
    bbox_labels:  根据groundtruth为每个anchor标注的偏移量 [batch, anchor_num * 4]
    bbox_masks:   根据groundtruth对对应背景的anchor进行mask [batch, anchor_num * 4]
    """
    # compute classification loss
    cls_loss = torch.tensor([0.0], dtype=torch.float32, device=device)
    bn = cls_preds.shape[0]
    for idx in range(bn):
        cls_loss += cls_criterion(cls_preds[idx], cls_labels[idx])
    cls_loss = cls_loss / bn

    # compute bbox regression loss
    bbox_loss = bbox_criterion(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    bbox_loss = bbox_loss * (bbox_masks.shape[1] / 4) 

    total_loss = cls_loss + bbox_loss
    return total_loss, cls_loss, bbox_loss 


# 为了测试刚刚定义的loss函数，我们需要计算出这5个tensor
# cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks
# 其中模型预测信息 cls_preds 和 bbox_preds 可以通过模型前向推理得到
# 标注信息 cls_labels, bbox_labels, bbox_masks 可以通过`锚框`一节介绍的 d2l.MultiBoxTarget 函数计算得到
# 为了复用 d2l.MultiBoxTarget 函数，我们需要对PIKACHU数据集迭代器返回的信息进行一些修改进行适配:

# 获取1个batch的数据
img, loc_chw, label = next(iter(train_loader)) 
# d2l.MultiBoxTarget 所需的类别信息不需考虑背景，因此label这里需要-1
label = label - 1
# pikachu数据集返回的bbox坐标为[cx,cy,w,h]格式，为了适配MultiBoxTarget函数变换为[x1,y1,x2,y2]
loc = d2l.center_2_hw(loc_chw)
# 将坐标+类别 组合成 ground truth 信息
ground_truth = torch.cat([label.unsqueeze(1), loc], dim=1)
# ground_truth的shape应为 [batch, true_box_num, 5],由于人造pikachu数据集每个图片仅有一个ground truth bbox,因此手动加出一个维度
ground_truth = ground_truth.unsqueeze(1)
ground_truth = ground_truth.float()

# 前向推理
img = img.to(device)
anchors, cls_preds, bbox_preds = net(img)

# 为每个锚框标注类别和偏移量
bbox_labels, bbox_masks, cls_labels = d2l.MultiBoxTarget(anchors.unsqueeze(0), ground_truth)
bbox_labels = bbox_labels.to(device)
bbox_masks = bbox_masks.to(device)
cls_labels = cls_labels.to(device)
print("bbox_labels shape: ", bbox_labels.shape)
print("bbox_masks shape: ", bbox_masks.shape)
print("cls_labels shape: ", cls_labels.shape)

# 测试下定义的损失函数和评价函数
total_loss, cls_loss, bbox_loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
print(total_loss, cls_loss, bbox_loss)


# 训练
num_epochs = 25
epoch = 0
for epoch in range(num_epochs):
        
    start_time = time.time()

    net.train()
    
    train_loss_sum = 0.0
    cls_loss_sum = 0.0
    loc_loss_sum = 0.0

    #for i, (img, loc_chw, label) in enumerate(train_loader):
    batch_idx = 0
    for img, loc_chw, label in train_loader:

        # 前向推理
        img = img.to(device)
        anchors, cls_preds, bbox_preds = net(img)

        # 适配d2l.MultiBoxTarget函数需要的groundtruth格式
        # d2l.MultiBoxTarget 所需的类别信息不需考虑背景，因此label这里需要-1
        label = label - 1
        # pikachu数据集返回的bbox坐标为[cx,cy,w,h]格式，为了适配MultiBoxTarget函数变换为[x1,y1,x2,y2]
        loc = d2l.center_2_hw(loc_chw)
        # 将坐标+类别 组合成 ground truth 信息
        ground_truth = torch.cat([label.unsqueeze(1), loc], dim=1)
        # ground_truth的shape应为 [batch, true_box_num, 5],由于人造pikachu数据集每个图片仅有一个ground truth bbox,因此手动加出一个维度
        ground_truth = ground_truth.unsqueeze(1)
        ground_truth = ground_truth.float()
        
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.MultiBoxTarget(anchors.unsqueeze(0), ground_truth)
        bbox_labels = bbox_labels.to(device)
        bbox_masks = bbox_masks.to(device)
        cls_labels = cls_labels.to(device)

        # compute loss
        total_loss, cls_loss, bbox_loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        train_loss_sum += total_loss.detach().cpu().item()
        cls_loss_sum += cls_loss.detach().cpu().item()
        loc_loss_sum += bbox_loss.detach().cpu().item()

        batch_idx += 1
        

    print('epoch %d, loss %.6f, cls loss %.6f, loc loss %.6f, time %.1f sec'
          % (epoch + 1, train_loss_sum/batch_idx, cls_loss_sum/batch_idx, loc_loss_sum/batch_idx, time.time() - start_time)) 
    print('-----------------------------------------------------------------')


# Saving the final Model
#checkpoints_save_dir = './ssd_outputs'
#d2l.save(net, checkpoints_save_dir, epoch, optimizer, train_loss_sum/batch_idx)


# Prediction
img = np.array(Image.open('pikachu.jpg').convert('RGB').resize((256, 256), Image.BILINEAR))
X = transforms.Compose([transforms.ToTensor()])(img).to(device)
X = X.to(device)

def predict(X, confidence_threshold=0.3):
    anchors, cls_preds, loc_preds = net(X.unsqueeze(0))
    #anchors = anchors.to(device)
    cls_preds = cls_preds.to(cpu_device)
    loc_preds = loc_preds.to(cpu_device)
    cls_preds = torch.nn.functional.softmax(cls_preds, dim=2)  # softmax操作
    cls_preds = cls_preds.permute(0,2,1)  # 维度调整，匹配d2l.MultiBoxDetection函数的输入要求

    # 整合网络推理结果(anchor及对应的类别预测和偏移量预测)得到最终的bbox预测结果 
    batch_bbox_info = d2l.MultiBoxDetection(cls_preds, loc_preds, anchors.unsqueeze(0), nms_threshold=0.5)

    # 解析batch_bbox_info，拿到非背景且置信度高于阈值的预测框
    # batch_bbox_info: shape (bn, 锚框个数, 6)
    # 每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
    # class_id=-1 表示背景或在非极大值抑制中被移除了
    bbox_idx = [i for i, bbox_info in enumerate(batch_bbox_info[0]) if bbox_info[0] != -1 and bbox_info[1] > confidence_threshold]

    return batch_bbox_info[0, bbox_idx]

bbox_info_predict = predict(X)
print(bbox_info_predict.shape)
print(bbox_info_predict)


def display(img, output):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    h, w = img.shape[0:2]
    bbox_scale = torch.tensor([w, h, w, h], dtype=torch.float32)
    bbox_list = []
    labels_list = []
    for row in output:
        bbox_list.append(row[2:6] * bbox_scale)
        labels_list.append('%.2f' % row[1])
    d2l.show_bboxes(fig.axes, bbox_list, labels_list, 'w')
    plt.savefig('pikachu_detect.png')

display(img, bbox_info_predict)
