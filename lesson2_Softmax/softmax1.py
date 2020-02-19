# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:40:07 2020

softmax和分类模型

softmax回归模型的从零开始实现，实现一个对Fashion-MNIST训练集中的图像数据进行分类的模型

@author: 伯禹教育
@modified by: as
"""
# import needed package
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

print(torch.__version__)
print(torchvision.__version__)

dataset_root = '../dataset'

print('-------------------------------------')
print('测试如何获取Fashion-MNIST数据集和读取数据')
print('-------------------------------------')
# get dataset, will auto download when the first run
mnist_train = torchvision.datasets.FashionMNIST(root=dataset_root, train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root=dataset_root, train=False, download=True, transform=transforms.ToTensor())

# show result 
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 我们可以通过下标来访问任意一个样本
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width

# 如果不做变换,得到的是PIL.Image格式的图像
mnist_PIL = torchvision.datasets.FashionMNIST(root=dataset_root, train=True, download=True)
print(type(mnist_train))
PIL_feature, label = mnist_PIL[0]
print(PIL_feature)

# define label map
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 读取数据测试
batch_size = 256
num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))



print('-------------------------------------')
print('softmax从零开始的实现')
print('-------------------------------------')

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

# 获取训练集数据和测试集数据
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, root=dataset_root)


# 模型参数初始化
num_inputs = 784  # 28 * 28 本练习为了处理图像,强行将图像数据摊平成一个向量
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


print('练习：对多维Tensor按维度操作')
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))  # dim为0，按照相同的列求和，并在结果中保留列特征
print(X.sum(dim=1, keepdim=True))  # dim为1，按照相同的行求和，并在结果中保留行特征
print(X.sum(dim=0, keepdim=False)) # dim为0，按照相同的列求和，不在结果中保留列特征
print(X.sum(dim=1, keepdim=False)) # dim为1，按照相同的行求和，不在结果中保留行特征

# 定义softmax操作
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # print("X size is ", X_exp.size())
    # print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制

print('测试softmax')
X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, '\n', X_prob.sum(dim=1))


# 定义softmax回归模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

print('练习：gather函数')
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
print(y_hat.gather(1, y.view(-1, 1)))
y = torch.LongTensor([0, 1, 0 ,1])
print(y_hat.gather(1, y.view(-1, 2)))


# 定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

print('练习：argmax函数')
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([2,0])
print(y_hat.argmax(dim=1))
print(y_hat.argmax(dim=0))
acc = y_hat.argmax(dim=1) == y
print(acc)
acc = acc.float().mean()
print(acc)
acc = acc.item()
print(acc)

# 定义准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
print("test evaluate_accuracy")
print(evaluate_accuracy(test_iter, net))


print('-------------------------------------')
print('训练')
print('-------------------------------------')
num_epochs, lr = 5, 0.1

def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的
    # 因为一般用PyTorch计算loss时就默认是沿batch维求平均,而不是sum。
    # 这个无大碍，根据实际情况即可
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data


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

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)


