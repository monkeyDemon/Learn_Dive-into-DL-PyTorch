# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 19:56:17 2020

VOC数据集相关练习

@author: as
"""
import sys
sys.path.append("../../../")
import d2lzh_pytorch as d2l

import os
import xml.dom.minidom
from matplotlib import pyplot as plt


# 定义xml文件解析函数
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


# --------------------------------------------
# 测试xml文件解析函数
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


# --------------------------------------------
# 可视化voc数据集
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


