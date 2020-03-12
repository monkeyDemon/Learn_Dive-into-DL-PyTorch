# datawhale 活动
import jieba
import numpy as np
# 结巴分词转化为list
# seg_list = list(jieba.cut("小明2019年毕业于清华大学",cut_all=False))
# print(seg_list)
# for i in seg_list:
#     print(i)
# 构建onehot编码
def set_of_words2vec(vocab_list, input_set):
    """
    遍历查看该单词是否出现，出现该单词则将该单词置1
    :param vocab_list: 所有单词集合列表
    :param input_set: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    result = [0] * len(vocab_list)
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in input_set:
        if word in vocab_list:
            result[vocab_list.index(word)] = 1
        else:
            # 这个后面应该注释掉，因为对你没什么用，这只是为了辅助调试的
            # print('the word: {} is not in my vocabulary'.format(word))
            pass
    return result
# 测试
def classify_naive_bayes(vec2classify, p0vec, p1vec, p_class1):
    """
    使用算法：
        # 将乘法转换为加法
        乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param vec2classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param p_class1: 类别1，侮辱性文件的出现概率
    :return: 类别1 or 0
    """
    # 计算公式  log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    # 我的理解是：这里的 vec2Classify * p1Vec 的意思就是将每个词与其对应的概率相关联起来
    # 可以理解为 1.单词在词汇表中的条件下，文件是good 类别的概率 也可以理解为 2.在整个空间下，文件既在词汇表中又是good类别的概率
    p1 = np.sum(vec2classify * p1vec) + np.log(p_class1)
    p0 = np.sum(vec2classify * p0vec) + np.log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0

# 训练朴素贝叶斯算法
def train_naive_bayes(train_mat, train_category):
    """
    朴素贝叶斯分类修正版，　注意和原来的对比，为什么这么做可以查看书
    :param train_mat:  type is ndarray
                    总的输入文本，大致是 [[0,1,0,1], [], []]
    :param train_category: 文件对应的类别分类， [0, 1, 0],
                            列表的长度应该等于上面那个输入文本的长度
    :return:
    """
    train_doc_num = len(train_mat)
    words_num = len(train_mat[0])
    # 因为侮辱性的被标记为了1， 所以只要把他们相加就可以得到侮辱性的有多少
    # 侮辱性文件的出现概率，即train_category中所有的1的个数，
    # 代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率
    print(len(train_category))
    pos_abusive = np.sum(train_category) / train_doc_num
    # 单词出现的次数
    # 原版，变成ones是修改版，这是为了防止数字过小溢出
    # p0num = np.zeros(words_num)
    # p1num = np.zeros(words_num)
    p0num = np.ones(words_num)
    p1num = np.ones(words_num)
    # 整个数据集单词出现的次数（原来是0，后面改成2了）
    p0num_all = 2.0
    p1num_all = 2.0

    for i in range(train_doc_num):
        # 遍历所有的文件，如果是侮辱性文件，就计算此侮辱性文件中出现的侮辱性单词的个数
        if train_category[i] == 1:
            p1num += train_mat[i]
            p1num_all += np.sum(train_mat[i])
        else:
            p0num += train_mat[i]
            p0num_all += np.sum(train_mat[i])
    # 后面改成取 log 函数
    p1vec = np.log(p1num / p1num_all)
    p0vec = np.log(p0num / p0num_all)
    return p0vec, p1vec, pos_abusive
def spanTest():
    '''

    :return:
    '''
    with open('train_shuffle.txt', 'r') as f:
        traindata_Num = f.readlines()
    train_label = []#训练标签
    train_data = []# 训练数据集合
    test_data = []#测试数据集合
    arr = []
    doc_list = []# 储存所有句子准备one hot编码
    # print(traindata_Num)
    for i in traindata_Num:
        if i == '\n':
            continue
        arr = i.replace('\n','').split('\t')
        # print(len(arr))
        train_list = list(jieba.cut(arr[1]))
        train_data.append(train_list)
        doc_list.append(train_list)
        # print(type(arr[0]))
        train_label.append(int(arr[0]))
    with open('test_handout.txt','r') as f:
        testdata_Num = f.readlines()
    for i in testdata_Num:
        if i == '\n':
            continue
        arr = i.replace('\n','')
        # print(len(arr))
        test_list = list(jieba.cut(arr))
        test_data.append(test_list)
        doc_list.append(train_list)
    # print(train_data)
    # print(test_data)
    # 创建词汇表
    vocab_list = create_vocab_list(doc_list)#创建一个词汇表
    # print(vocab_list)'凤爪', '简餐', '门市部', '健谈', '早饭'
    # 训练数据
    # 构建训练矩阵:
    # print(train_data[0])
    training_mat = []
    training_class = []
    for i in range(len(train_data)):
        training_mat.append(set_of_words2vec(vocab_list=vocab_list,input_set=train_data[i]))# 构建训练矩阵
        training_class.append(train_label[i])
        # print(train_data[i])
    p0v,p1v,p_spam = train_naive_bayes(training_mat,training_class)
    result = []
    for i in  range(len(test_data)):
        #测试数据
        result.append(classify_naive_bayes(np.array(set_of_words2vec(vocab_list=vocab_list,input_set=test_data[i])),p0v,p1v,p_spam))
    # print(result)
    import csv
    with open('submission.csv', 'w') as f:
        f.write('ID,Prediction\n')
        for i in range(len(result)):
            f.write('{},{}\n'.format(i,float(result[i])))
        # 这里不需要readlines

# 创建一个单词集合
def create_vocab_list(data_set):
    """
    获取所有单词的集合
    :param data_set: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    """
    vocab_set = set()  # create empty set
    for item in data_set:
        # | 求两个集合的并集
        vocab_set = vocab_set | set(item)
    return list(vocab_set)
spanTest()