# -*- coding: utf-8 -*-
'''
Created on 2020/2/26 14:04

@Author  : Xyao

@File    : fine_tuning_Xyao.py

@result  : very bad
'''

import os
import sys
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import models
from d2lzh_pytorch import utils
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置可见GPU设备号；
                                          # 当拥有多个GPU设备时，可以合理分配GPU资源

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("进行数据集和预训练权重文件检查")
pretrain_dir = '../../pretrain/resnet18'
resnet18_weight_path = os.path.join(pretrain_dir, 'resnet18-5c106cde.pth')
if not os.path.exists(resnet18_weight_path):
    print("预训练权重文件不存在{}".format(resnet18_weight_path))
    raise RuntimeError("please check first")

dataset_dir = '../../dataset/FashionMNIST'
if not os.path.exists(dataset_dir):
    print("数据集文件不存在{}".format(resnet18_weight_path))
    raise RuntimeError("please check first")
print('所需文件均已存在')

# 定义加载数据集的函数
def load_data_fashion_mnist(batch_size, root='../../dataset', use_normalize=False, mean=None, std=None):
    """Download the fashion mnist dataset and then load into memory."""

    if use_normalize:
        normalize = transforms.Normalize(mean=[mean], std=[std])
        '''
        train_augs = transforms.Compose([transforms.RandomCrop(28, padding=2),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize])
        '''
        train_augs = transforms.Compose([transforms.Resize(size=224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize])
        test_augs = transforms.Compose([transforms.Resize(size=224),
                                        transforms.ToTensor(),
                                        normalize])
    else:
        train_augs = transforms.Compose([transforms.ToTensor()])
        test_augs = transforms.Compose([transforms.ToTensor()])

    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=train_augs)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=test_augs)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

'''
展示一个批次的图像以及其对应的标签
print('Show some samples.')
batch_size = 5
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='../../dataset', use_normalize=False)
for Xdata,ylabel in train_iter:
    break
X, y = [], []
for i in range(5):
    print(Xdata[i].shape,ylabel[i].numpy())
    X.append(Xdata[i]) # 将第i个feature加到X中
    y.append(ylabel[i].numpy()) # 将第i个label加到y中

_, figs = plt.subplots(1, len(X), figsize=(12, 12))
for f, img, lbl in zip(figs, X, y):
   f.imshow(img.view((28, 28)).numpy())
   f.set_title(lbl)
   f.axes.get_xaxis().set_visible(False)
   f.axes.get_yaxis().set_visible(False)
plt.show()
'''
'''
# 原始图片的形状为[1, 28, 28]
# 通过torch.cat()，扩充通道数，将其形状变为[3, 28, 28] (同一张图片在通道维度上拼接)
print('展示拼接后形状是否正确')
batch_size = 5
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='../../dataset', use_normalize=False)
for Xdata,ylabel in train_iter:
    break

for i in range(5):
    print(Xdata[i].shape, ylabel[i].numpy())
    X = torch.cat([Xdata[i], Xdata[i], Xdata[i]], 0)
    print(X.shape)
'''
# 首次获取数据集(用于计算整个数据集的均值和标准差)
print('计算数据集均值标准差')
batch_size = 64  
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='../../dataset', use_normalize=False)
# 求整个数据集的均值
temp_sum = 0
cnt = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break   # 最后一个batch不足batch_size,这里就忽略了
    channel_mean = torch.mean(X, dim=(0,2,3))  # 按channel求均值(不过这里只有1个channel)
    cnt += 1   # cnt记录的是batch的个数，不是图像
    temp_sum += channel_mean[0].item()
dataset_global_mean = temp_sum / cnt
print('整个数据集的像素均值:{}'.format(dataset_global_mean))
# 求整个数据集的标准差
cnt = 0
temp_sum = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break   # 最后一个batch不足batch_size,这里就忽略了
    residual = (X - dataset_global_mean) ** 2
    channel_var_mean = torch.mean(residual, dim=(0,2,3))  
    cnt += 1   # cnt记录的是batch的个数，不是图像
    temp_sum += math.sqrt(channel_var_mean[0].item())
dataset_global_std = temp_sum / cnt
print('整个数据集的像素标准差:{}'.format(dataset_global_std))

# 重新获取应用了归一化的数据集迭代器
print('应用均值标准差对数据进行归一化')
batch_size = 32
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='../../dataset', use_normalize=True,
                        mean = dataset_global_mean, std = dataset_global_std)


# 加载官方resnet18实现
pretrained_net = models.resnet18(pretrained=False)
pretrained_net.load_state_dict(torch.load(resnet18_weight_path))

print('打印原始预训练模型的全连接层')
print(pretrained_net.fc)

print('重新定义全连接层')
pretrained_net.fc = nn.Linear(512, 10)
print(pretrained_net.fc)

# 获取全连接层权重对应的内存id
output_params = list(map(id, pretrained_net.fc.parameters()))
# 获取除全连接层以外的模型参数
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())



def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    net.eval()
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            n += y.shape[0]
    net.train() # 改回训练模式
    return acc_sum / n

def train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_test_acc = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.4f, test acc %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        if test_acc > best_test_acc:
            print('find best! save at model/best.pth')
            best_test_acc = test_acc
            torch.save(net.state_dict(), 'model/best.pth')

print('开始训练')

pre_Conv_net = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1, stride=1),
                             nn.ReLU())

scratch_net = models.resnet18(pretrained=False, num_classes=10)

net = nn.Sequential(pre_Conv_net, scratch_net)

lr = 0.001
num_epochs = 50
# 将随机初始化的fc layer学习率设为已经预训练过的部分的10倍
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pre_Conv_net.parameters(), 'lr': lr * 10},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, momentum=0.9, weight_decay=5e-4)
train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
