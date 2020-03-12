# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:23:26 2020

数据增强

@author: 伯禹教育
@modified by: as
"""
import os
import sys
import time
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision

sys.path.append("../")

import d2lzh_pytorch as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)
