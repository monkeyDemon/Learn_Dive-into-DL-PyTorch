# -*- coding: utf-8 -*-
'''
Created on 2020/2/27 16:23

@Author  : Xyao

@File    : GoogleNet_Try.py
'''
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import d2lzh_pytorch as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # TODO:


class GlobalAvgPool2d(nn.Module):
    """
    全局平均池化层
    可通过将普通的平均池化的窗口形状设置成输入的高和宽实现
    """
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        """
            use_1×1conv: 是否使用额外的1x1卷积层来修改通道数
            stride: 卷积层的步幅, resnet使用步长为2的卷积来替代pooling的作用，是个很赞的idea
        """
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    '''
    resnet block

    num_residuals: 当前block包含多少个残差块
    first_block: 是否为第一个block

    一个resnet block由num_residuals个残差块组成
    其中第一个残差块起到了通道数的转换和pooling的作用
    后面的若干残差块就是完成正常的特征提取
    '''
    if first_block:
        assert in_channels == out_channels # 第一个模块的输出通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

# 定义Inception块
class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出


net = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),   #
                   nn.BatchNorm2d(32),
                   nn.ReLU())

net.add_module("resnet_block1", resnet_block(32, 32, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(32, 64, 2))
net.add_module("inception1", Inception(64, 32, (32, 64), (16, 32), 8))
net.add_module("maxpool1", nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))
net.add_module("inception2", Inception(136, 68, (68, 128), (32, 64), 16))
net.add_module("maxpool2", nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)))
net.add_module("global_avg_pool", GlobalAvgPool2d())
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(276, 10)))


print('打印网络结构')
print(net)

print('打印 1*1*28*28 输入经过每个模块后的shape')
X = torch.rand(1, 1, 28, 28)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)
# net.children() 获取一级子模块
# net.named_children()获取一级子模块及其名字, 这里我没并没有手动指定名字，因此打印出的是编号


# 定义加载数据集的函数
def load_data_fashion_mnist(batch_size, root='../../dataset', use_normalize=False, mean=None, std=None):
    """Download the fashion mnist dataset and then load into memory."""

    if use_normalize:
        normalize = transforms.Normalize(mean=[mean], std=[std])
        train_augs = transforms.Compose([transforms.RandomCrop(28, padding=2),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize])
        test_augs = transforms.Compose([transforms.ToTensor(), normalize])
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


print('计算数据集均值标准差')
batch_size = 64
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='../../dataset', use_normalize=False)
# 求整个数据集的均值
temp_sum = 0
cnt = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break  # 最后一个batch不足batch_size,这里就忽略了
    channel_mean = torch.mean(X, dim=(0, 2, 3))  # 按channel求均值(不过这里只有1个channel)
    cnt += 1  # cnt记录的是batch的个数，不是图像
    temp_sum += channel_mean[0].item()
dataset_global_mean = temp_sum / cnt
print('整个数据集的像素均值:{}'.format(dataset_global_mean))
# 求整个数据集的标准差
cnt = 0
temp_sum = 0
for X, y in train_iter:
    if y.shape[0] != batch_size:
        break  # 最后一个batch不足batch_size,这里就忽略了
    residual = (X - dataset_global_mean) ** 2
    channel_var_mean = torch.mean(residual, dim=(0, 2, 3))
    cnt += 1  # cnt记录的是batch的个数，不是图像
    temp_sum += math.sqrt(channel_var_mean[0].item())
dataset_global_std = temp_sum / cnt
print('整个数据集的像素标准差:{}'.format(dataset_global_std))

# 重新获取应用了归一化的数据集迭代器
batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, root='../../dataset', use_normalize=True,
                                                mean=dataset_global_mean, std=dataset_global_std)


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
    net.train()  # 改回训练模式
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
            # utils.save_model({
            #    'arch': args.model,
            #    'state_dict': net.state_dict()
            # }, 'saved-models/{}-run-{}.pth.tar'.format(args.model, run))


print('训练...')
lr, num_epochs = 0.01, 100
optimizer = optim.Adam(net.parameters(), lr=lr)
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # TODO:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_model(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

print('加载最优模型')
net.load_state_dict(torch.load('model/best.pth'))
net = net.to(device)

print('inference测试集')
net.eval()
id = 0
preds_list = []
with torch.no_grad():
    for X, y in test_iter:
        batch_pred = list(net(X.to(device)).argmax(dim=1).cpu().numpy())
        for y_pred in batch_pred:
            preds_list.append((id, y_pred))
            id += 1

print('生成提交结果文件')
with open('submission.csv', 'w') as f:
    f.write('ID,Prediction\n')
    for id, pred in preds_list:
        f.write('{},{}\n'.format(id, pred))
