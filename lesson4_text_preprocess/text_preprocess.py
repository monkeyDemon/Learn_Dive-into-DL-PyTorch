# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:22:11 2020

文本预处理

本代码将介绍文本数据的常见预处理步骤：
读入文本
分词
建立字典，将每个词映射到一个唯一的索引（index）
将文本从词的序列转换为索引的序列，方便输入模型

@author: 伯禹教育
@modified by: as
"""

print('---------------------------------------')
print('读入文本')
print('---------------------------------------')
import collections
import re

def read_time_machine():
    with open('timemachine.txt', 'r') as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
    return lines
# strip去除首尾指定元素，默认\n和空格
# ltrip去除首元素，rstrip去除末尾元素


lines = read_time_machine()
print('# sentences %d' % len(lines))
print('6-10 line:')
for line in lines[5:10]:
    print(line)


print('---------------------------------------')
print('分词')
print('---------------------------------------')
def tokenize(sentences, token='word'):
    """Split sentences into word or char tokens"""
    if token == 'word':
        return [sentence.split(' ') for sentence in sentences]
    elif token == 'char':
        return [list(sentence) for sentence in sentences]
    else:
        print('ERROR: unkown token type '+token)

tokens = tokenize(lines)
print('6-10 line(分词后):')
for token_list in tokens[5:10]:
    print(token_list)


print('---------------------------------------')
print('建立字典')
print('---------------------------------------')

class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        counter = count_corpus(tokens)  # : 
        self.token_freqs = list(counter.items())   # [('the', 2261), ...]
        self.idx_to_token = []
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['', '', '', '']
        else:
            self.unk = 0
            self.idx_to_token += ['']
        self.idx_to_token += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(sentences):
    tokens = [tk for st in sentences for tk in st]
    return collections.Counter(tokens)  # 返回一个字典，记录每个词的出现次数

vocab = Vocab(tokens)
print(type(vocab.token_to_idx.items()))
print(list(vocab.token_to_idx.items())[0:10])
print(vocab['time'])
print(vocab.to_tokens(2))


print('---------------------------------------')
print('将词转换为索引')
print('---------------------------------------')
for i in range(8, 10):
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])


print('---------------------------------------')
print('用现有工具进行分词')
print('---------------------------------------')
# 我们前面介绍的分词方式非常简单，它至少有以下几个缺点:
#
# 1. 标点符号通常可以提供语义信息，但是我们的方法直接将其丢弃了
# 2. 类似“shouldn't", "doesn't"这样的词会被错误地处理
# 3. 类似"Mr.", "Dr."这样的词会被错误地处理
# 我们可以通过引入更复杂的规则来解决这些问题，但是事实上，有一些现有的工具可以很好地进行分词，我们在这里简单介绍其中的两个：spaCy和NLTK。

text = "Mr. Chen doesn't agree with my suggestion."


#print('使用spaCy进行分词')
#import spacy
#nlp = spacy.load('en_core_web_sm')
#doc = nlp(text)
#print([token.text for token in doc])


print('使用nltk进行分词')
from nltk.tokenize import word_tokenize
from nltk import data
data.path.append('../dataset/nltk_data')
print(word_tokenize(text))
