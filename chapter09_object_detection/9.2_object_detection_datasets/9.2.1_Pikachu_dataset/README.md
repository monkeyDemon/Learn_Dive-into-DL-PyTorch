# 9.2.1 目标检测数据集（皮卡丘）

在目标检测领域并没有类似MNIST或Fashion-MNIST那样的小数据集。为了快速测试模型，我们可以合成了一个小的数据集。

除了构造一个小的便于测试的数据集之外，通过本小节还可以了解如何自定义一个数据集DataLoader，后面的章节并不会使用本节生成的数据集，因此如果你对这部分内容不感兴趣，可以选择跳过。

![皮卡丘](https://s1.ax1x.com/2020/03/13/8KPtot.jpg)

我们首先使用一个开源的皮卡丘3D模型生成了1,000张不同角度和大小的皮卡丘图像。然后我们收集了一系列背景图像，并在每张图的随机位置放置一张随机的皮卡丘图像。

原书使用MXNet提供的工具将图像转换成二进制的RecordIO格式。该格式既可以降低数据集在磁盘上的存储开销，又能提高读取效率，这里我们不需要详细了解。

因此为了获得皮卡求数据集的原始文件，我们需要安装MXNet，装cpu版本即可，生成完数据集就可以卸载了

```
pip install mxnet
```

## 1. 获取数据集

首先引入必要的包
```
import sys
sys.path.append("../../../")
import d2lzh_pytorch as d2l

import os
import json
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms

pikachu_dataset = '../../../dataset/pikachu'
```

RecordIO格式的皮卡丘数据集可以直接在网上下载。获取数据集的操作定义在download_pikachu函数中。

```python
def download_pikachu(data_dir):
    root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
                'gluon/dataset/pikachu/')
    dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
               'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
               'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
    for k, v in dataset.items():
        d2l.download_url(root_url + k, data_dir)
```

与原书内容不同的是，这里我们为了后面使用Pytorch训练目标检测模型时更方便，需要将原书中提供的数据格式转换回我们更熟悉的png图片形式。

由于转换部分的代码与目标检测的学习关系并不大，具体的细节就省略了，感兴趣的同学详见`d2l.download_and_preprocess_pikachu_data`函数。

调用如下函数，一键完成皮卡丘数据集的准备工作，将数据处理成熟悉的png格式的图片和json格式的标注文件

```python
d2l.download_and_preprocess_pikachu_data(dir=pikachu_dataset)
```

## 2. 读取数据集

这里我们创建一个`PIKACHU`类（已经放入d2l中），用来加载前面生成的标注文件（包含了图片名称和对应的目标框信息的json）。然后我们即可创建一个pytorch的DataLoader来使用PIKACHU类完成训练过程的数据读取工作。

```python
# We created a class PIKACHU which loads the annotation file (json) 
# which contains image names along with their bounding box annotations. 
# 为方便后面使用，我们将PIKACHU封装到d2l中
class PIKACHU(torch.utils.data.Dataset):
    def __init__(self, data_dir, set, transform=None, target_transform=None):

        self.image_size = (3, 256, 256)
        self.images_dir = os.path.join(data_dir, set, 'images')

        self.set = set
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        self.target_transform = target_transform

        annotations_file = os.path.join(data_dir, set, 'annotations.json')
        with open(annotations_file) as file:
            self.annotations = json.load(file)

    def __getitem__(self, index):

        annotations_i = self.annotations['data_' + str(index+1)]

        image_path = os.path.join(self.images_dir, annotations_i['image'])
        img = np.array(Image.open(image_path).convert('RGB').resize((self.image_size[2], self.image_size[1]), Image.BILINEAR))
        # print(img.shape)
        loc = np.array(annotations_i['loc'])

        label = 1 - annotations_i['class']

        if self.transform is not None:
            img = self.transform(img)
        return (img, loc, label)

    def __len__(self):
        return len(self.annotations)
```

下面我们读取一个小批量并打印图像和标签的形状。

```python
train_dataset = d2l.PIKACHU(pikachu_dataset, 'train')
val_dataset = d2l.PIKACHU(pikachu_dataset, 'val')

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=4)
batch = next(iter(train_loader))
print(batch[0].shape, batch[1].shape, batch[2].shape)
```

输出结果：

```
torch.Size([16, 3, 256, 256]) torch.Size([16, 4]) torch.Size([16])
```

## 3. 图示数据

我们首先定义可视化目标框相关的函数

```python
# 本函数已保存在dd2lzh_pytorch包中方便以后使用
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format.
    Convert the bounding box (top-left x, top-left y, bottom-right x, bottom-right y) 
    format to matplotlib format: ((upper-left x, upper-left y), width, height)
    """
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
                        fill=False, edgecolor=color, linewidth=2)
  
# 本函数已保存在dd2lzh_pytorch包中方便以后使用
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes.
    bboxes: 待绘制的bbox， need be format as [[x1,y1,x2,y2],[...], ..., [...]]
    labels: 与要绘制的bbox一一对应的标注信息，将会绘制在bbox的左上角
    colors: 标注框显示的颜色，不设置会自动使用几个默认颜色进行轮换
    """ 
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox, color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=6, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

# 本函数已保存在dd2lzh_pytorch包中方便以后使用
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes
```

我们画出10张图像和它们中的边界框。

可以看到，皮卡丘的角度、大小和位置在每张图像中都不一样。当然，这是一个简单的人工数据集。实际中的数据通常会复杂得多。

```python
imgs = [train_dataset[i][0].permute(1,2,0) for i in range(10)]
labels = [d2l.center_2_hw(torch.Tensor(train_dataset[i][1]).unsqueeze(0)) for i in range(10)]

show_num_rows = 2
show_num_cols = 5
axes = d2l.show_images(imgs, show_num_rows, show_num_cols, scale=2)

for i in range(show_num_rows):
    for j in range(show_num_cols):
        index = i * show_num_cols + j
        ax = axes[i][j]
        label = labels[index]
        show_bboxes(ax, [label.squeeze(0)*256], colors=['r'])
plt.savefig('visual_pikachu_dataset.png')
```

![可视化皮卡求数据集](https://s1.ax1x.com/2020/03/13/8KP1Qe.png)

## 4. 小结

合成的皮卡丘数据集可用于测试目标检测模型。

目标检测的数据读取与图像分类的类似，本小节展示了如何自定义一个数据集读取器

在引入边界框后，如果后续进行图像增广（如随机裁剪）会变得复杂，可以思考下如何编写代码。
