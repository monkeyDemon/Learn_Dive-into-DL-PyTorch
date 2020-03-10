# Learn_Dive-into-DL-PyTorch

本项目诞生于 Datawhale:whale:第10期组队学习活动：《动手学深度学习》Pytorch版, 由 Datawhale成员安晟维护

《动手学深度学习》是亚马逊首席科学家李沐等人编写的一本优秀的深度学习教学，原书作者：阿斯顿·张、李沐、扎卡里 C. 立顿、亚历山大 J. 斯莫拉以及其他社区贡献者

中文版：[动手学深度学习](https://zh.d2l.ai/) | [Github仓库](https://github.com/d2l-ai/d2l-zh)       

English Version: [Dive into Deep Learning](https://d2l.ai/) | [Github Repo](https://github.com/d2l-ai/d2l-en)

原书实现代码为MXNet，针对本书的中英两个版本的Pytorch重构，github有两个版本：
[中文版Pytorch重构](https://github.com/ShusenTang/Dive-into-DL-PyTorch), [英文版Pytorch重构](https://github.com/dsgiitr/d2l-pytorch)

本项目正在对以上优质资源的代码进行学习和复现，后期将会力求进一步扩展，补充最新的模型，训练trick，学术进展等

持续更新中...

视频学习资料见伯禹[动手学深度学习课程页面](https://www.boyuai.com/elites/course/cZu18YmweLv10OeV)



# baseline汇总 :rocket:

## CV大作业 

# 0.9625分方案

这是为帮助大家学习，我本人在比赛过程中就开源的baseline

适合初学者一点点探索，附带教程，介绍了如何将得分一步步从 0.9235-> 0.9432 -> 0.9526

**我线下训出的最好成绩是0.9625，尽快整理出来分享给大家**

详见目录：`assignment1_FinshionMNIST_Classification`

# 0.9621分(第一名解决方案:beer:)

感谢`啊啦啦啦`同学分享的解决方案,使用了很多比较新的数据增强方案，值得学习~ [代码](https://github.com/whut2962575697/image_classification)

## NLP大作业

# 0.95340分方案

感谢[艾春辉](https://blog.csdn.net/weixin_42479155)同学提供的95.340分的baseline

主要是给没有基础的小伙伴一个指引，包括如何下载数据集，保存模型，生成提交结果的一个简单流程。

这个baseline的代码已经整理到了本项目中，代码详见`assignment2_Recommended_Comments`

# 0.96965分方案

感谢 `sun1106645759` 同学分享的方案 

[博客](https://www.jianshu.com/p/4d28479d2f44)   [代码](https://github.com/guangweis/BERT--Recommended-comments)

# 0.7101分(第一名解决方案:beer:)

感谢`gbl555`同学分享的解决方案

[博客+代码](https://blog.csdn.net/G_B_L/article/details/104619341)


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


## 目录与代码更新进度
* 阅读指南
* 1\. 深度学习简介
* 2\. 预备知识
   - [ ] 2.1 环境配置
   - [ ] 2.2 数据操作
   - [ ] 2.3 自动求梯度
* 3\. 深度学习基础
   - [ ] 3.1 线性回归
   - [ ] 3.2 线性回归的从零开始实现
   - [ ] 3.3 线性回归的简洁实现
   - [ ] 3.4 softmax回归
   - [ ] 3.5 图像分类数据集（Fashion-MNIST）
   - [ ] 3.6 softmax回归的从零开始实现
   - [ ] 3.7 softmax回归的简洁实现
   - [ ] 3.8 多层感知机
   - [ ] 3.9 多层感知机的从零开始实现
   - [ ] 3.10 多层感知机的简洁实现
   - [ ] 3.11 模型选择、欠拟合和过拟合
   - [ ] 3.12 权重衰减
   - [ ] 3.13 丢弃法
   - [ ] 3.14 正向传播、反向传播和计算图
   - [ ] 3.15 数值稳定性和模型初始化
   - [ ] 3.16 实战Kaggle比赛：房价预测
* 4\. 深度学习计算
   - [ ] 4.1 模型构造
   - [ ] 4.2 模型参数的访问、初始化和共享
   - [ ] 4.3 模型参数的延后初始化
   - [ ] 4.4 自定义层
   - [ ] 4.5 读取和存储
   - [ ] 4.6 GPU计算
* 5\. 卷积神经网络
   - [ ] 5.1 二维卷积层
   - [ ] 5.2 填充和步幅
   - [ ] 5.3 多输入通道和多输出通道
   - [ ] 5.4 池化层
   - [ ] 5.5 卷积神经网络（LeNet）
   - [ ] 5.6 深度卷积神经网络（AlexNet）
   - [ ] 5.7 使用重复元素的网络（VGG）
   - [ ] 5.8 网络中的网络（NiN）
   - [ ] 5.9 含并行连结的网络（GoogLeNet）
   - [ ] 5.10 批量归一化
   - [ ] 5.11 残差网络（ResNet）
   - [ ] 5.12 稠密连接网络（DenseNet）
* 6\. 循环神经网络
   - [ ] 6.1 语言模型
   - [ ] 6.2 循环神经网络
   - [ ] 6.3 语言模型数据集（周杰伦专辑歌词）
   - [ ] 6.4 循环神经网络的从零开始实现
   - [ ] 6.5 循环神经网络的简洁实现
   - [ ] 6.6 通过时间反向传播
   - [ ] 6.7 门控循环单元（GRU）
   - [ ] 6.8 长短期记忆（LSTM）
   - [ ] 6.9 深度循环神经网络
   - [ ] 6.10 双向循环神经网络
* 7\. 优化算法
   - [ ] 7.1 优化与深度学习
   - [ ] 7.2 梯度下降和随机梯度下降
   - [ ] 7.3 小批量随机梯度下降
   - [ ] 7.4 动量法
   - [ ] 7.5 AdaGrad算法
   - [ ] 7.6 RMSProp算法
   - [ ] 7.7 AdaDelta算法
   - [ ] 7.8 Adam算法
* 8\. 计算性能
   - [ ] 8.1 命令式和符号式混合编程
   - [ ] 8.2 异步计算
   - [ ] 8.3 自动并行计算
   - [ ] 8.4 多GPU计算
* 9\. 计算机视觉
   - [ ] 9.1 图像增广
   - [ ] 9.2 微调
   - [ ] 9.3 目标检测和边界框
   - [ ] 9.4 锚框
   - [ ] 9.5 多尺度目标检测
   - [ ] 9.6 目标检测数据集（皮卡丘）
   - [ ] 9.7 单发多框检测（SSD）
   - [ ] 9.8 区域卷积神经网络（R-CNN）系列
   - [ ] 9.9 语义分割和数据集
   - [ ] 9.10 全卷积网络（FCN）
   - [ ] 9.11 样式迁移
   - [ ] 9.12 实战Kaggle比赛：图像分类（CIFAR-10）
   - [ ] 9.13 实战Kaggle比赛：狗的品种识别（ImageNet Dogs）
* 10\. 自然语言处理
   - [ ] 10.1 词嵌入（word2vec）
   - [ ] 10.2 近似训练
   - [ ] 10.3 word2vec的实现
   - [ ] 10.4 子词嵌入（fastText）
   - [ ] 10.5 全局向量的词嵌入（GloVe）
   - [ ] 10.6 求近义词和类比词
   - [ ] 10.7 文本情感分类：使用循环神经网络
   - [ ] 10.8 文本情感分类：使用卷积神经网络（textCNN）
   - [ ] 10.9 编码器—解码器（seq2seq）
   - [ ] 10.10 束搜索
   - [ ] 10.11 注意力机制
   - [ ] 10.12 机器翻译



持续更新中......



## 引用
如果您在研究中使用了这个项目请引用原书:
```
@book{zhang2019dive,
    title={Dive into Deep Learning},
    author={Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola},
    note={\url{http://www.d2l.ai}},
    year={2020}
}
```


