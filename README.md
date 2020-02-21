# Learn_Dive-into-DL-PyTorch

Datawhale第10期组队学习活动：《动手学深度学习》Pytorch版 的练习代码

课程页面：https://www.boyuai.com/elites/course/cZu18YmweLv10OeV

本项目由5群助教维护, 为在自己环境进行练习的同学提供一个参考。

本项目也会根据课程进度，上传必要的数据集，方便大家下载。

学习中常见问题及解决办法：https://shimo.im/docs/86tr6VvQVRdvkX8r

## 大作业baseline 新!:rocket:

为方便大家学习，这里给出一个比较基本的baseline，得分0.9235

主要是给没有基础的小伙伴一个指引，包括如何下载数据集，保存模型，生成提交结果的一个简单流程。

详见目录： `assignment1_FinshionMNIST_Classification`

## 如何下载数据集到自己的环境？

按照官方给出的解释，work文件夹下的数据都能下载到本地，input文件夹里的数据原则上不能下载

所以我们用代码把数据集挪到work下就好了:tada:

![download-dataset](https://raw.githubusercontent.com/monkeyDemon/Learn_Dive-into-DL-PyTorch/master/imgs/download_dataset.jpg)

图片有时加载不出来，这里再简单描述下步骤：

步骤1:在镜像中添加一个代码块，加入拷贝input目录下数据集到work目录下，并进行打包的代码

步骤2:运行代码段

步骤3:右键点击work下的数据集文件并进行下载

步骤4:在本地解压(和鲸提供的镜像没装zip命令，所以我只能压成tar的，若无法解压得装一个解压软件，经测试好压是可以)

需要根据实际情况替换下面的路径，代码示例：

``` python
import os
cp_str = 'cp -r /home/kesci/input/nltk_data3784/nltk_data /home/kesci/work'
os.system(cp_str)
tar_str = 'tar czvf /home/kesci/work/nltk_data.tar /home/kesci/work/nltk_data'
os.system(tar_str)
print('finish')
```


## 学习内容安排与代码更新进度

* 线性回归 :beer:

* softmax回归 :beer:

* 多层感知机 :beer:

* 文本预处理 :beer:

* 语言模型 :beer:

* 循环神经网络基础

* 过拟合、欠拟合及其解决方案

* 梯度消失、梯度爆炸

* 循环神经网络进阶

* 机器翻译及相关技术

* 注意力机制与Seq2seq模型

* Transformer

* 卷积神经网络基础 :beer:

* leNet :beer:

* 卷积神经网络进阶 :beer:

* 批量归一化和残差网络 :beer:

* 凸优化

* 梯度下降

* 优化算法进阶

* word2vec

* 词嵌入进阶

* 文本分类

* 数据增强

* 模型微调

* 目标检测基础

* 图像风格迁移

* 图像分类案例1

* 图像分类案例2

* GAN

* DCGAN

* 代码大作业


## 【奖励机制与要求】

基本奖励：结营证书（完成学习任务和打卡）

Datawhale奖励：【优秀学习者】和【优秀团队】证书，Datawhale【组织邀请】（优质学习笔记，群内积极讨论，帮助答疑）

额外奖励：【保密】：（伯禹平台讨论区优秀问题，学习成果，参与答疑）



## 【打卡规则】

1.本次学习的打卡形式为自行选择平台（CSDN，简书，Github等）撰写【学习笔记】，学习结束后助教将根据打卡内容，评选【优秀学习者】和【优秀团队】并【颁发证书】。

2.本次学习一共5次打卡，每位同学需在打卡截止期前打卡，一次不打卡将会【被抱出群】（依然可以在伯禹平台继续学习）

3.优秀的学习成果可通过打卡结果参考，更希望大家沉淀在伯禹平台讨论区，触发【额外奖励】。


## 【参考资料】

《动手学深度学习》中文版官网教材：http://zh.gluon.ai/ 

PyTorch中文文档：https://pytorch-cn.readthedocs.io/zh/stable/

部分PyTorch代码来自GitHub开源仓库：https://github.com/ShusenTang/Dive-into-DL-PyTorch
