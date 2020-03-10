# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 23:58:19 2020

训练EfficientNet

运行此代码，你需要先按照pytorch版EfficientNet
see: https://github.com/lukemelas/EfficientNet-PyTorch

@author: as
"""
import os
import sys
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models

from efficientnet_pytorch import EfficientNet


os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # TODO:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("进行数据集文件检查")
dataset_dir = '../dataset/hotdog'
if not os.path.exists(dataset_dir):
    print("数据集文件不存在{}".format(resnet18_weight_path))
    raise RuntimeError("please check first")


print('定义加载模块...')
train_imgs = ImageFolder(os.path.join(dataset_dir, 'train'))
test_imgs = ImageFolder(os.path.join(dataset_dir, 'test'))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])



print('加载 efficientnet模型...')
net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)

print('打印 1*3*224*224 输入经过模型后得到的特征的shape')
X = torch.rand((1, 3, 224, 224))
features = net.extract_features(X)
print(features.shape) # torch.Size([1, 1280, 7, 7])


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

def train_model(net, train_iter, test_iter, optimizer, device, num_epochs, lr, lr_period, lr_decay):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_test_acc = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        if epoch > 0 and epoch % lr_period == 0:  # 每lr_period个epoch，学习率衰减一次
            lr = lr * lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
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


print('-----------------------------')
print('训练：微调模型')
lr = 0.001
lr_period = 4
lr_decay = 0.1
num_epochs = 200
batch_size = 32
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
train_iter = DataLoader(ImageFolder(os.path.join(dataset_dir, 'train'), transform=train_augs),
                        batch_size, shuffle=True, num_workers=4)
test_iter = DataLoader(ImageFolder(os.path.join(dataset_dir, 'test'), transform=test_augs),
                       batch_size, num_workers=4)
net = net.to(device)
train_model(net, train_iter, test_iter, optimizer, device, num_epochs, lr, lr_period, lr_decay)

