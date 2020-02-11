# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:58:17 2020

线性回归模型从零开始的实现

@author: 伯禹教育
"""

# ----------------------------------
# 矢量运算优势的小实验
# ----------------------------------
import torch
import time
# init variable a, b as 1000 dimension vector
n = 1000
a = torch.ones(n)
b = torch.ones(n)

# define a timer class to record time
class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        # start the timer
        self.start_time = time.time()

    def stop(self):
        # stop the timer and record time into a list
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        # calculate the average and return
        return sum(self.times)/len(self.times)

    def sum(self):
        # return the sum of recorded time
        return sum(self.times)

timer = Timer()
c = torch.zeros(n)
for i in range(n):
    c[i] = a[i] + b[i]
print('%.5f sec' % timer.stop())


timer.start()
d = a + b
print('%.5f sec' % timer.stop())



# ----------------------------------
# 生成数据集
# ----------------------------------
# import packages and modules
import torch
import random
import numpy as np
from matplotlib import pyplot as plt

print(torch.__version__)

# set input feature number 
num_inputs = 2
# set example number
num_examples = 1000

# set true weight and bias in order to generate corresponded label
true_w = [2, -3.4]
true_b = 4.2

features = torch.randn(num_examples, num_inputs,
                      dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)


# ----------------------------------
# 读取数据集
# ----------------------------------
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # random read 10 samples
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # the last time may be not enough for a whole batch
        yield  features.index_select(0, j), labels.index_select(0, j)

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


# ----------------------------------
# 初始化模型参数
# 线性回归模型 y = wx + b，参数只有两个张量 w 和 b
# ----------------------------------
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

w.requires_grad_(requires_grad=True)   # 手动设置是否需要计算梯度
b.requires_grad_(requires_grad=True)


# ----------------------------------
# 定义模型
# ----------------------------------
def linreg(X, w, b):
    return torch.mm(X, w) + b       # 注意区分 torch.mul 点乘 和 torch.mm 矩阵乘法 的区别

# ----------------------------------
# 定义损失函数
# ----------------------------------
def squared_loss(y_hat, y): 
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

# ----------------------------------
# 定义优化函数
# ----------------------------------
def sgd(params, lr, batch_size): 
    for param in params:
        param.data -= lr * param.grad / batch_size # ues .data to operate param without gradient track

# ----------------------------------
# 训练流程
# ----------------------------------
# super parameters init
lr = 0.03
num_epochs = 5

net = linreg
loss = squared_loss

# training
for epoch in range(num_epochs):  # training repeats num_epochs times
    # in each epoch, all the samples in dataset will be used once
    
    # X is the feature and y is the label of a batch sample
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  
        # calculate the gradient of batch sample loss 
        l.backward()  
        # using small batch random gradient descent to iter model parameters
        sgd([w, b], lr, batch_size)  
        # reset parameter gradient
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
