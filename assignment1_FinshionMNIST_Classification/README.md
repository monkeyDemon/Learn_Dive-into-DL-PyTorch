# CV 大作业：Fashion-mnist分类任务

本次任务需要针对Fashion-MNIST数据集，设计、搭建、训练机器学习模型，能够尽可能准确地分辨出测试数据地标签。

最后提交一个csv文件，格式如下：

|  ID   | Prediction  |
|  ----  | ----  |
| 0  | 4(预测类别) |
| 1  | 9 |
| 2  | 3 |

## baseline 0.9235

为方便大家学习，这里给出一个比较基本的baseline，得分0.9235

主要是给没有基础的小伙伴一个指引，包括如何下载数据集，保存模型，生成提交结果的一个简单流程。

详见目录：`baseline`

## 2020/2/22更新 0.9432:rocket:

天又做了一些简单改动，将分数提到了0.9432

主要是3点改动：输入归一化、数据增强、sgd

详见目录：`baseline_plus`


## 参考文献

[1] Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747

[2] https://github.com/zalandoresearch/fashion-mnist/
