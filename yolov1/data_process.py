# 这个文件用来对VOC2012进行数据预处理

import xml.etree.ElementTree as ET
import cv2
import os

# 分类类别导入，该表为类别和索引的转换表
GL_CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
              'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
              'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

GL_NUMBBOX = 2  # bbox数量
GL_NUMGRID = 7  # grid cell数量

STATIC_DATA_PATH = 'D:/Codefield/py/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'  # 数据集文件夹
STATIC_LABEL_PATH = 'labels'  # 标注.txt文件夹
STATIC_IMAGE_PATH = 'JPEGImages'  # 图片所在文件夹
STATIC_DEBUG = False  # 调试用


# 将boundbox中的左上角右下角点的坐标形式，转换为中心点+wh的格式，并进行归一化
def convert(size, box):
    # 求wh的归一化分母
    dw = 1. / size[0]
    dh = 1. / size[1]
    # xy坐标转换
    x = (box[1] + box[0]) / 2.0
    y = (box[3] + box[2]) / 2.0
    # wh坐标转换
    w = box[1] - box[0]
    h = box[3] - box[2]
    # 归一化
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


# convert_annotation()函数：读取Annotations文件夹下的每一个xml文件并调用convert()函数
# voc2012数据集中的标注文件存放在xml文件下，使用elementtree读取xml文件
# 需要把xml文件中的坐标转换为x、y、w、h
def convert_annotation(anno_dir, image_id, labels_dir):
    # 打开标注数据集所在文件夹，标注数据集在annotation文件夹中
    in_file = open(os.path.join(anno_dir, 'Annotations/%s' % image_id))
    # 获得图片的准确id
    image_id = image_id.split('.')[0]
    # 处理xml文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    # 获取图片长宽的信息
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        # 判断是否是困难图片
        if obj.find('difficult'):
            difficult = int(obj.find('difficult').text)
        else:
            difficult = 0

        cls = obj.find('name').text
        if cls not in GL_CLASSES or int(difficult) == 1:
            continue
        cls_id = GL_CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        # 获得标注坐标点
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                  float(xmlbox.find('ymax').text))
        bb = convert((w, h), points)  # 返回(x,y,w,h)
        with open(os.path.join(os.path.join(anno_dir, labels_dir), '%s.txt' % image_id), 'a') as out_file:
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def make_label_txt(anno_dir, labels_dir):
    # 在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息
    filenames = os.listdir(os.path.join(anno_dir, 'Annotations'))
    for file in filenames:
        convert_annotation(anno_dir, file, labels_dir)


def show_labels_img(imgname):
    """imgname是输入图像的名称，无下标"""
    img = cv2.imread(STATIC_DATA_PATH + "/JPEGImages/" + imgname + ".jpg")
    h, w = img.shape[:2]
    print(w, h)
    label = []
    with open("D:/Codefield/py/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/labels/" + imgname + ".txt", 'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]
            print(GL_CLASSES[int(label[0])])
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))
            print(label)
            print(pt1)
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))
            print(pt2)
            cv2.putText(img, GL_CLASSES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.rectangle(img, pt1, pt2, (0, 0, 255, 2))

    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # make_label_txt(STATIC_DATA_PATH, STATIC_LABEL_PATH)
    show_labels_img('2008_000008')
