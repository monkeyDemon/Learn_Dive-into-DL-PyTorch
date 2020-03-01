# NLP 大作业：推荐评论展示任务

任务描述：本次推荐评论展示任务的目标是从真实的用户评论中，挖掘合适作为推荐理由的短句。

数据集：本次推荐评论展示任务所采用的数据集是点评软件中，用户中文评论的集合。

数据集文件分为训练集和测试集部分，对应文件如下：

带标签的训练数据：train_shuffle.txt

不带标签的测试数据：test_handout.txt

更多的题目细节见文件`推荐评论展示任务.ipynb`

下面是收集到的不同方案的baseline，大家可以参考着学习。

代码我暂时没有经过确认，所以路径等变量可能需要自行调整


## 朴素贝叶斯

0.85分左右，详见`naive_bayes`

感谢[艾春辉](https://blog.csdn.net/weixin_42479155)同学提供的baseline

## bi-lstm

95.340:rocket:

代码详见`bi-lstm`

感谢[艾春辉](https://blog.csdn.net/weixin_42479155)同学提供的baseline
