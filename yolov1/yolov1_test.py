import numpy as np
import torch
from matplotlib.patches import draw_bbox
from torch.utils.data import DataLoader
import cv2
from LOSS import calculate_iou
from data_set import VOC2012, GL_CLASSES


# 注意检查一下输入数据的格式，到底是xywh还是xyxy
def labels2bbox(matrix):
    """
    将网络输出的7*7*30的数据转换为bbox的(98,25)的格式，然后再将NMS处理后的结果返回
    :param matrix: 注意，输入的数据中，bbox坐标的格式是(px,py,w,h)，需要转换为(x1,y1,x2,y2)的格式再输入NMS
    :return: 返回NMS处理后的结果
    """
    if matrix.size()[0:2] != (7, 7):
        raise ValueError("Error: Wrong labels size:", matrix.size())
    bbox = torch.zeros((98, 25))
    # 先把7*7*30的数据转变为bbox的(98,30)的格式，其中，bbox信息格式从(px,py,w,h)转换为(x1,y1,x2,y2),方便计算iou
    for i in range(7):  # i是网格的行方向(y方向)
        for j in range(7):  # j是网格的列方向(x方向)
            bbox[2 * (i * 7 + j), 0:4] = torch.Tensor([(matrix[i, j, 0] + j) / 7 - matrix[i, j, 2] / 2,
                                                       (matrix[i, j, 1] + i) / 7 - matrix[i, j, 3] / 2,
                                                       (matrix[i, j, 0] + j) / 7 + matrix[i, j, 2] / 2,
                                                       (matrix[i, j, 1] + i) / 7 + matrix[i, j, 3] / 2])
            bbox[2 * (i * 7 + j), 4] = matrix[i, j, 4]
            bbox[2 * (i * 7 + j), 5:] = matrix[i, j, 10:]
            bbox[2 * (i * 7 + j) + 1, 0:4] = torch.Tensor([(matrix[i, j, 5] + j) / 7 - matrix[i, j, 7] / 2,
                                                           (matrix[i, j, 6] + i) / 7 - matrix[i, j, 8] / 2,
                                                           (matrix[i, j, 5] + j) / 7 + matrix[i, j, 7] / 2,
                                                           (matrix[i, j, 6] + i) / 7 + matrix[i, j, 8] / 2])
            bbox[2 * (i * 7 + j) + 1, 4] = matrix[i, j, 9]
            bbox[2 * (i * 7 + j) + 1, 5:] = matrix[i, j, 10:]
    print(bbox)
    return NMS(bbox)  # 对所有98个bbox执行NMS算法，清理cls-specific confidence score较低以及iou重合度过高的bbox


def NMS(bbox, conf_thresh=0.1, iou_thresh=0.3):
    """bbox数据格式是(n,25),前4个是(x1,y1,x2,y2)的坐标信息，第5个是置信度，后20个是类别概率
    :param conf_thresh: cls-specific confidence score的阈值
    :param iou_thresh: NMS算法中iou的阈值
    """
    n = bbox.size()[0]
    bbox_prob = bbox[:, 5:].clone()  # 类别预测的条件概率
    print(bbox_prob)
    bbox_confi = bbox[:, 4].clone().unsqueeze(1).expand_as(bbox_prob)  # 预测置信度
    bbox_cls_spec_conf = bbox_confi * bbox_prob  # 置信度*类别条件概率=cls-specific confidence score整合了是否有物体及是什么物体的两种信息
    bbox_cls_spec_conf[bbox_cls_spec_conf <= conf_thresh] = 0  # 将低于阈值的bbox忽略
    for c in range(20):
        rank = torch.sort(bbox_cls_spec_conf[:, c], descending=True).indices
        for i in range(98):
            if bbox_cls_spec_conf[rank[i], c] != 0:
                for j in range(i + 1, 98):
                    if bbox_cls_spec_conf[rank[j], c] != 0:
                        iou = calculate_iou(bbox[rank[i], 0:4], bbox[rank[j], 0:4])
                        if iou > iou_thresh:  # 根据iou进行非极大值抑制抑制
                            bbox_cls_spec_conf[rank[j], c] = 0
    bbox = bbox[torch.max(bbox_cls_spec_conf, dim=1).values > 0]  # 将20个类别中最大的cls-specific confidence score为0的bbox都排除
    bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(bbox_cls_spec_conf, dim=1).values > 0]
    res = torch.ones((bbox.size()[0], 6))
    res[:, 1:5] = bbox[:, 0:4]  # 储存最后的bbox坐标信息
    res[:, 0] = torch.argmax(bbox[:, 5:], dim=1).int()  # 储存bbox对应的类别信息
    res[:, 5] = torch.max(bbox_cls_spec_conf, dim=1).values  # 储存bbox对应的class-specific confidence scores
    return res


def show_labels_img(image, flabel):
    h, w = image.shape[:2]
    print(w, h)
    print(image.shape)
    for label in flabel:
        print(GL_CLASSES[int(label[0])])
        print(label)
        pt1 = int(label[1] * w), int(label[2] * h)
        pt2 = int(label[3] * w), int(label[4] * h)
        cv2.putText(image, GL_CLASSES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.rectangle(image, pt1, pt2, (0, 0, 255, 2))

    cv2.imshow("img", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    val_dataloader = DataLoader(VOC2012(is_train=True), batch_size=1, shuffle=False)
    model = torch.load("./models_pkl/YOLOv1_epoch50.pkl")  # 加载训练好的模型
    for i, (inputs, labels1) in enumerate(val_dataloader):
        inputs = inputs.cuda()
        # 以下代码是测试labels2bbox函数的时候再用
        # labels = labels1.float().cuda()
        # labels = labels.squeeze(dim=0)
        # labels = labels.permute((1, 2, 0))
        pred = model(inputs)  # pred的尺寸是(1,30,7,7)
        pred = pred.squeeze(dim=0)  # 压缩为(30,7,7)
        pred = pred.permute((1, 2, 0))  # 转换为(7,7,30)

        # 测试labels2bbox时，使用 labels作为labels2bbox2函数的输入
        bbox = labels2bbox(pred)
        # 此处可以用labels代替pred，测试一下输出的bbox是否和标签一样，从而检查labels2bbox函数是否正确。当然，还要注意将数据集改成训练集而不是测试集，因为测试集没有labels。
        inputs = inputs.squeeze(dim=0)  # 输入图像的尺寸是(1,3,448,448),压缩为(3,448,448)
        inputs = inputs.permute((1, 2, 0))  # 转换为(448,448,3)
        img = inputs.cpu().numpy()
        img = 255 * img  # 将图像的数值从(0,1)映射到(0,255)并转为非负整形
        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img)
        show_labels_img(img, bbox)
        input()
