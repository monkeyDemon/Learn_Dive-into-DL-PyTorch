# 9.2.2 PASCAL VOC 数据集

PASCAL VOC 挑战赛是视觉领域的赛事，包含多个子任务，其中以目标检测任务最为著名。我们熟知的`VOC2007`和`VOC2012`就是来自该竞赛。

[Pascal VOC网址](http://host.robots.ox.ac.uk/pascal/VOC/) 与 [排行榜](http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php)

[PASCAL VOC 2007 挑战赛主页](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)、[PASCAL VOC 2012 挑战赛主页](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

我们通常使用VOC数据集来验证目标检测算法的有效性(另一个标准数据集是COCO，我们将会再后面的小节介绍)。此外，VOC数据集还被作为一种标准的目标检测数据集格式来看待，因此了解VOC数据集的结构对我们使用开源项目训练自己的目标检测任务是非常必要的。

## 1. VOC数据集下载

进入`dataset`目录，可以看到`VOC2007.sh`和`VOC2012.sh`两个脚本

```
dataset$ ./VOC2007.sh
```

运行脚本即可一键完成相应数据集的下载

## 2. 数据集分析

### 2.1 目标类别

PASCAL VOC 2007 和 2012 数据集共20个类别（算背景21类），分布是：

aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor

### 2.2 目录结构

我们观察下载的VOC2007数据集目录：

```
VOCdevkit
├──VOC2007
    ├── Annotations 进行 detection 任务时的标签文件，xml 形式，文件名与图片名一一对应
    ├── JPEGImages 存放 .jpg 格式的图片文件
    ├── SegmentationClass 存放按照语义分割的图片
    ├── SegmentationObject 存放按照实例分割的图片
    ├── ImageSets 
        ├── Layout
        ├── Segmentation
        ├── Main 存放的是分类和检测的数据集分割文件
        │   ├── train.txt 写着用于训练的图片名称， 共 2501 个
        │   ├── val.txt 写着用于验证的图片名称，共 2510 个
        │   ├── trainval.txt train与val的合集。共 5011 个
        │   ├── test.txt 写着用于测试的图片名称，共 4952 个
```

### 2.3 XML文件解析

VOC数据集的XML标注文件有着统一的文件结构：[VOC2011 Annotation Guidelines](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/guidelines.html)

我们观察`0000001.xml`来直观的看一下xml文件的内容

```xml
<annotation>
    <folder>VOC2007</folder>
    <filename>000001.jpg</filename>  # 文件名 
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>341012865</flickrid>
    </source>
    <owner>
        <flickrid>Fried Camels</flickrid>
        <name>Jinky the Fruit Bat</name>
    </owner>
    <size>  # 图像尺寸, 用于对 bbox 左上和右下坐标点做归一化操作
        <width>353</width>
        <height>500</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>  # 是否用于分割
    <object>
        <name>dog</name>  # 物体类别
        <pose>Left</pose>  # 拍摄角度：front, rear, left, right, unspecified 
        <truncated>1</truncated>  # 目标是否被截断（比如在图片之外），或者被遮挡（超过15%）
        <difficult>0</difficult>  # 检测难易程度，这个主要是根据目标的大小，光照变化，图片质量来判断
        <bndbox>
            <xmin>48</xmin>
            <ymin>240</ymin>
            <xmax>195</xmax>
            <ymax>371</ymax>
        </bndbox>
    </object>
    <object>
        <name>person</name>
        <pose>Left</pose>
        <truncated>1</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>8</xmin>
            <ymin>12</ymin>
            <xmax>352</xmax>
            <ymax>498</ymax>
        </bndbox>
    </object>
</annotation>
```

了解了xml的结构，我们尝试编写如下代码来读取xml文件

首先导入必要的库
```python
import sys
sys.path.append("../../../")
import d2lzh_pytorch as d2l

import os
import xml.dom.minidom
from matplotlib import pyplot as plt
```

定义xml文件解析函数
```python
def parse_voc_xml(xml_path):
    """解析voc格式的xml数据标注文件
    返回：
        object_name_list: 包含图片中物体类别信息的列表，与bbox_list一一对应
        bbox_lsit: 包含图片中物体坐标信息的列表
    """
    DomTree = xml.dom.minidom.parse(xml_path)
    annotation = DomTree.documentElement

    filename_list = annotation.getElementsByTagName('filename') 
    filename = filename_list[0].childNodes[0].data
    object_list = annotation.getElementsByTagName('object')
    
    object_name_list = []
    bbox_list = []
    for object_ele in object_list:
        object_name = object_ele.getElementsByTagName('name')[0].childNodes[0].data
        bbox_ele = object_ele.getElementsByTagName('bndbox')[0]
        x1 = bbox_ele.getElementsByTagName('xmin')[0].childNodes[0].data
        y1 = bbox_ele.getElementsByTagName('ymin')[0].childNodes[0].data
        x2 = bbox_ele.getElementsByTagName('xmax')[0].childNodes[0].data
        y2 = bbox_ele.getElementsByTagName('ymax')[0].childNodes[0].data
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        object_name_list.append(object_name)
        bbox_list.append(bbox)
    return object_name_list, bbox_list
```

测试下xml文件解析函数，尝试解析遍历VOC数据集，并读取第一个xml文件
```python
VOC_dir = '../../../dataset/VOCdevkit/VOC2007'
img_dir = os.path.join(VOC_dir, 'JPEGImages')
anno_dir = os.path.join(VOC_dir, 'Annotations')
name_list_path = os.path.join(VOC_dir, 'ImageSets/Main/train.txt')

file_name_list = []
with open(name_list_path, 'r') as reader:
    for line in reader.readlines():
        file_name = line.rstrip()
        file_name_list.append(file_name)

for file_name in file_name_list:
    img_name = file_name + '.jpg'
    xml_name = file_name + '.xml'
    img_path = os.path.join(img_dir, img_name)
    xml_path = os.path.join(anno_dir, xml_name)
    object_name_list, bbox_list = parse_voc_xml(xml_path)
    print(object_name_list)
    print(bbox_list)
    break
```

解析成功，获取到的类别信息和坐标信息为：
```
['car']
[[156, 97, 351, 270]]
```

接下来，我们可视化出VOC数据集中的部分图片，保存在`visual_voc`目录下
```python
voc_visual_dir = 'visual_voc'
visual_cnt = 50

visual_cnt = min(visual_cnt, len(file_name_list))
if not os.path.exists(voc_visual_dir):
    os.mkdir(voc_visual_dir)
for file_name in file_name_list[:visual_cnt]:
    img_name = file_name + '.jpg'
    xml_name = file_name + '.xml'
    img_path = os.path.join(img_dir, img_name)
    xml_path = os.path.join(anno_dir, xml_name)
    visual_img_save_path = os.path.join(voc_visual_dir, img_name)

    object_name_list, bbox_list = parse_voc_xml(xml_path)

    img = plt.imread(img_path)
    fig = plt.imshow(img)
    d2l.show_bboxes(fig.axes, bbox_list, object_name_list)
    plt.savefig(visual_img_save_path)
    plt.close()
```

运行成功，VOC数据集部分图片的可视化结果如下：

![voc_show](https://s1.ax1x.com/2020/03/31/GlmGhn.png)

## 3. VOC格式数据集标注工具

了解了VOC数据集的格式，理论上我们就可以将任何形式的目标检测标注数据转化为VOC的格式了。

当然，如果你还没有标注数据，这里推荐一个标注工具`labelImg`

这个工具可以比较方便的进行目标检测数据的标注，并且帮你生成VOC标准格式的数据集文件。
