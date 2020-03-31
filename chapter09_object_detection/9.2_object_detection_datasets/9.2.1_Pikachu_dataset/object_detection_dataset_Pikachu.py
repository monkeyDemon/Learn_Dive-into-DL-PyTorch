# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 19:39:32 2020

目标检测数据集：皮卡丘~

本脚本我们制作一个皮卡丘数据集，用于快速测试目标检测模型的效果

首先使用一个开源的皮卡丘3D模型生成了1,000张不同角度和大小的皮卡丘图像。
然后我们收集了一系列背景图像，并在每张图的随机位置放置一张随机的皮卡丘图像。

为了使用这个数据集，你需要安装mxnet，cpu版本即可
pip install mxnet

@author: as
"""
import sys
sys.path.append("../../../")
import d2lzh_pytorch as d2l

import os
import json
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms


pikachu_dataset = '../../../dataset/pikachu'

# RecordIO格式的皮卡丘数据集可以直接在网上下载。获取数据集的操作定义在download_pikachu函数中。
def download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        d2l.download_url(root_url + k, data_dir)

# 下载原始格式的数据集
#download_pikachu(data_dir=pikachu_dataset)

# Convert Dataset to .PNG Images
# 由于涉及到比较多细节，为了更加方便，下载和预处理数据的代码整理到了如下函数中
# 调用如下函数，一键完成皮卡丘数据集的准备工作，将数据处理成后续我们可以用torch处理的png格式的图片
d2l.download_and_preprocess_pikachu_data(dir=pikachu_dataset)


# Read the Dataset
# We created a class PIKACHU which loads the annotation file (json) 
# which contains image names along with their bounding box annotations. 
# Create Dataloaders in Pytorch 
class PIKACHU(torch.utils.data.Dataset):
    def __init__(self, data_dir, set, transform=None, target_transform=None):
        
        self.image_size = (3, 256, 256)
        self.images_dir = os.path.join(data_dir, set, 'images')

        self.set = set
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.target_transform = target_transform

        annotations_file = os.path.join(data_dir, set, 'annotations.json')
        with open(annotations_file) as file:
            self.annotations = json.load(file)

    def __getitem__(self, index):
        
        annotations_i = self.annotations['data_' + str(index+1)]
        
        image_path = os.path.join(self.images_dir, annotations_i['image'])
        img = np.array(Image.open(image_path).convert('RGB').resize((self.image_size[2], self.image_size[1]), Image.BILINEAR))
        # print(img.shape)
        loc = np.array(annotations_i['loc'])
        
        label = 1 - annotations_i['class']
    
        if self.transform is not None:
            img = self.transform(img)       
        return (img, loc, label)

    def __len__(self):
        return len(self.annotations)



train_dataset = d2l.PIKACHU(pikachu_dataset, 'train')
val_dataset = d2l.PIKACHU(pikachu_dataset, 'val')

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=4)
batch = next(iter(train_loader))
print(batch[0].shape, batch[1].shape, batch[2].shape)


# Graphic Data

# 本函数已保存在dd2lzh_pytorch包中方便以后使用
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format.
    Convert the bounding box (top-left x, top-left y, bottom-right x, bottom-right y) 
    format to matplotlib format: ((upper-left x, upper-left y), width, height)
    """
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
                        fill=False, edgecolor=color, linewidth=2)
  
# 本函数已保存在dd2lzh_pytorch包中方便以后使用
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes.
    bboxes: 待绘制的bbox， need be format as [[x1,y1,x2,y2],[...], ..., [...]]
    labels: 与要绘制的bbox一一对应的标注信息，将会绘制在bbox的左上角
    colors: 标注框显示的颜色，不设置会自动使用几个默认颜色进行轮换
    """ 
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox, color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=6, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

# 本函数已保存在dd2lzh_pytorch包中方便以后使用
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


imgs = [train_dataset[i][0].permute(1,2,0) for i in range(10)]
labels = [torch.Tensor(train_dataset[i][1]).unsqueeze(0) for i in range(10)]

show_num_rows = 2
show_num_cols = 5
axes = d2l.show_images(imgs, show_num_rows, show_num_cols, scale=2)

for i in range(show_num_rows):
    for j in range(show_num_cols):
        index = i * show_num_cols + j
        ax = axes[i][j]
        label = labels[index]
        d2l.show_bboxes(ax, [label.squeeze(0)*256], colors=['r'])
plt.savefig('visual_pikachu_dataset.png') 
