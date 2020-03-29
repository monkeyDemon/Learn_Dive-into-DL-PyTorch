# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:58:17 2020

使用pytorch的线性回归简洁实现

@author: 伯禹教育
@modified by: as
"""

import torch
from torch import nn
import numpy as np
torch.manual_seed(1)

print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')


# ----------------------------------------------
# 生成数据集
# ----------------------------------------------
num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)


# ----------------------------------------------
# 读取数据集
# ----------------------------------------------
import torch.utils.data as Data

batch_size = 10

# combine featues and labels of dataset
dataset = Data.TensorDataset(features, labels)

# put dataset into DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # whether shuffle the data or not
    num_workers=2,              # read data in multithreading
)

for X, y in data_iter:
    print(X, '\n', y)
    break



# ----------------------------------------------
# 定义模型
# ----------------------------------------------
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()      # call father function to init 
        self.linear = nn.Linear(n_feature, 1)  # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`

    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net)


# ----------------------------------------------
# 定义多层网络的方法
# ----------------------------------------------
## ways to init a multilayer network
## method one
#net = nn.Sequential(
#    nn.Linear(num_inputs, 1)
#    # other layers can be added here
#    )
#
## method two
#net = nn.Sequential()
#net.add_module('linear', nn.Linear(num_inputs, 1))
## net.add_module ......
#
## method three
#from collections import OrderedDict
#net = nn.Sequential(OrderedDict([
#          ('linear', nn.Linear(num_inputs, 1))
#          # ......
#        ]))
#
#print(net)
#print(net[0])



# ----------------------------------------------
# 初始化模型参数 
# 如何初始化参数，需要根据上面如何定义网络来决定
# ----------------------------------------------
from torch.nn import init

for name in net.state_dict():
   print(name)
#linear.weight
#linear.bias
init.normal_(net.state_dict()['linear.weight'], mean=0.0, std=0.01)
init.constant_(net.state_dict()['linear.bias'], val=0.0)  # or you can use `net[0].bias.data.fill_(0)` to modify it directly

#init.normal_(net[0].weight, mean=0.0, std=0.01)
#init.constant_(net[0].bias, val=0.0)  # or you can use `net[0].bias.data.fill_(0)` to modify it directly

for param in net.parameters():
    print(param)


# ----------------------------------------------
# 定义损失函数和优化函数 
# ----------------------------------------------
loss = nn.MSELoss()    # nn built-in squared loss function
                       # function prototype: `torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')`

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)   # built-in random gradient descent function
print(optimizer)  # function prototype: `torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)`



# ----------------------------------------------
# 训练 
# ----------------------------------------------
num_epochs = 5
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # reset gradient, equal to net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))


