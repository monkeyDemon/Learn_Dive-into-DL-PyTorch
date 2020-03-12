# -*- coding:utf-8 -*- 
# 此代码在https://github.com/bamboosjtu/ai_learning/blob/master/finalwork.py的基础方改进而来，感谢作者贡献
# pip install torchtext
# 预训练词向量下载地址：https://github.com/shiyanlou-015555/Chinese-Word-Vectors

import collections
import os
import random
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据,借助了前辈的baseline
def read_comments(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            if line=='\n':
                continue
            label, comment = line.split()
            data.append([comment, int(label)])
    random.shuffle(data)
    return data
all_data = read_comments("/home/kesci/input/good1014/train_shuffle.txt")

# 提交的时候用全部数据训练，调参的时候要切分训练集和预测集
train_data = all_data[:16000]
valid_data = all_data[14000:]

# 预处理数据
def get_tokenized_comments(data):
    """
    data: list of [string, label]
    """
    def tokenizer(text):
        return list(text)#
        # 多次实验表明,不适用词向量,我们使用字向量可能更好
    return [tokenizer(comment) for comment, _ in data]

def get_vocab_comments(data):
    # print(data[:5])
    tokenized_data = get_tokenized_comments(data)
    # print(tokenized_data[:3])
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=2)

# 创建数据集
vocab = get_vocab_comments(train_data)
print(len(vocab))

# 构建词汇表成功
max_l = 50# 我们设定每个句子长度为30
def preprocess_comments(data, vocab):
    """因为每条评论长度不一致所以不能直接组合成小批量，我们定义preprocess_imdb函数对每条评论进行分词，并通过词典转换成词索引，然后通过截断或者补0来将每条评论长度固定成15。"""
      # 将每条评论通过截断或者补0，使得长度变成500
    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))
    tokenized_data = get_tokenized_comments(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels

batch_size = 64
train_set = Data.TensorDataset(*preprocess_comments(train_data, vocab))
valid_set = Data.TensorDataset(*preprocess_comments(valid_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
valid_iter = Data.DataLoader(valid_set, batch_size)

# 装载训练和测试数据
for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'#batches:', len(train_iter)

# RNN
class BiRNN(torch.nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               dropout=0.1,
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        # 改进
        # self.dropout = nn.Dropout(p=0.1)
        # self.decoder = nn.Linear(4*num_hiddens, 2)
        # 改进
        self.decoder = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4*num_hiddens, 2*num_hiddens),
            nn.Dropout(p=0.1),
            nn.Linear(4*num_hiddens,2)
            )
    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs

# 字符向量
embed_size, num_hiddens, num_layers = 300, 300, 2
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)# 创建网络
print(net)
# 使用预训练模型
# 导入模型
from gensim.models import KeyedVectors
w2v = KeyedVectors.load_word2vec_format('/home/kesci/input/good1014/sgns.wiki.word')
def load_pretrained_embedding(words, pretrained_vocab):
    '''
    @params:
        words: 需要加载词向量的词语列表，以 itos (index to string) 的词典形式给出
        pretrained_vocab: 预训练词向量
    @return:
        embed: 加载到的词向量
    '''
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为0
    oov_count = 0 # out of vocabulary
    # embed[0,:] = torch.tensor([0])
    for i, word in enumerate(words):
        # try:
        #     idx = pretrained_vocab.stoi[word]
        #     embed[i, :] = pretrained_vocab.vectors[idx]
        # except KeyError:
        #     oov_count += 1
        # print(word)
        # 对于不存在字典中的字,我们会给一个较低的函数变量
        if word not in pretrained_vocab:
            # print(word)
            embed[i,:] = torch.tensor(np.random.normal(0, 0.1, size=pretrained_vocab.vectors[0].shape[0]))
            #           dtype=torch.float32)
            #embed[i,:] = torch.tensor([0]*pretrained_vocab.vectors[0].shape[0])
        else:
            embed[i,:] = torch.Tensor(pretrained_vocab[word])
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    print(torch.tensor(np.random.normal(0, 0.1, size=pretrained_vocab.vectors[0].shape[0])))
    return embed
# print(load_pretrained_embedding(vocab.itos, w2v))
net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, w2v))
net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它
# 准确度评价指标
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n
# 使用自己定义的softmax
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition
#auc评价指标
def evaluate_auc(data_iter, net, device=None):
    if device is None:
        device = list(net.parameters())[0].device
    y_true, y_hat = np.zeros(0), np.zeros(0)
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            y_hat = np.concatenate([y_hat, softmax(net(X.to(device)).detach().cpu())[:,1].numpy()])
            y_true = np.concatenate([y_true, y.cpu().numpy()])
            net.train() # 改回训练模式
    return roc_auc_score(y_true, y_hat), y_hat
# 训练函数
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
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
        train_auc, _ = evaluate_auc(train_iter, net)
        test_acc = evaluate_accuracy(test_iter, net)
        test_auc, _ = evaluate_auc(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, train auc %.3f, test auc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, train_auc, test_auc, time.time() - start))
# 进行训练
lr, num_epochs = 0.01, 4
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
train(train_iter, valid_iter, net, loss, optimizer, device, num_epochs)
# 输出结果
lines =  []
with open("/home/kesci/input/good1014/test_handout.txt", encoding="utf-8") as f:
    for line in f:
        if line=='\n':
            continue
        lines.append(line)
print(lines[:5])

def preprocess_comments2(data, vocab):
    """因为每条评论长度不一致所以不能直接组合成小批量，我们定义preprocess_imdb函数对每条评论进行分词，并通过词典转换成词索引，然后通过截断或者补0来将每条评论长度固定成10。"""
    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))
    tokenized_data = [list(line.strip()) for line in data]
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    return features
test_X = preprocess_comments2(lines, vocab)
print(test_X[:3])
test_y = []
print(test_X[1])
for i in test_X:
    # print(softmax(net(i.unsqueeze(0).to(device))).detach().cpu()[:,1].numpy()[0])
    test_y.append(softmax(net(i.unsqueeze(0).to(device))).detach().cpu()[:,1].numpy()[0])
import csv
with open('submission.csv', 'w') as f:
    f.write('ID,Prediction\n')
    for i in range(len(test_y)):
        f.write('{},{}\n'.format(i,float(test_y[i])))
