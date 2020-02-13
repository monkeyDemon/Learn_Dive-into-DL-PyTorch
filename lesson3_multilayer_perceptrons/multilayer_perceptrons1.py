# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:45:32 2020

多层感知机

使用多层感知机图像分类的从零开始的实现

@author: 伯禹教育
@modified by: as
"""
import sys
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)


print('-----------------------------------------')
print('基本知识与语法练习')
print('-----------------------------------------')

print("练习几个新函数的使用：torch.arange, detach, relu")
print('arange')
x = torch.arange(-3.0, 3.0, 0.5, requires_grad=True)
print(x)

print('detach')
# detach 函数用于切断反向传播, 可参考 https://blog.csdn.net/weixin_34363171/article/details/94236818
x_detach = x.detach()
print(x_detach)

# numpy 作用在 Variable 上面时，不能有梯度，这时就要先用detach()的特性辅助一下，可以尝试运行下面两行
x_numpy = x.detach().numpy()
print(x_numpy)
# 下面的用法会报错
#x_numpy = x.numpy()
#print(x_numpy)

print('relu')
y = x.relu()
print(y)

print('backward')
y.sum().backward()  # 体会sum的作用，去掉是不行的
print(x.grad)

print('sigmoid')
y = x.sigmoid()
print(y)

print('sigmoid 的导数')
x.grad.zero_()
y.sum().backward()
print(x.grad)

print('tanh')
y = x.tanh()
print(y)

print('tanh 的导数')
x.grad.zero_()
y.sum().backward()
print(x.grad)



print('-----------------------------------------')
print('多层感知机从零开始的实现')
print('-----------------------------------------')

# 获取训练集
def load_data_fashion_mnist(batch_size, resize=None, root='../dataset'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans) # Compose函数用于组合多个图像变换
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

dataset_root = '../dataset'
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, root=dataset_root)


# 定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)


# 自定义relu激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))


# 定义一个三层bp神经网络
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


# 定义交叉熵损失函数
loss = torch.nn.CrossEntropyLoss()


print('-----------------------------------------')
print('训练')
print('-----------------------------------------')

def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的
    # 因为一般用PyTorch计算loss时就默认是沿batch维求平均,而不是sum。
    # 这个无大碍，根据实际情况即可
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

num_epochs, lr = 5, 100.0
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
