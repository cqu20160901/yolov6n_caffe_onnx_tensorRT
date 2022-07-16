#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import os.path as osp
import cv2
import torch
import numpy as np
from math import exp


caffe_root = '/root/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe

net_file = './yolov6n.prototxt'
caffe_model = './yolov6n.caffemodel'

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()

net = caffe.Net(net_file, caffe_model, caffe.TEST)


CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]


meshgrid = []


class_num = len(CLASSES)
headNum = 3
strides = [8, 16, 32]

mapSize = [[36, 64], [18, 32], [9, 16]]
nmsThresh = 0.51
objectThresh = 0.45

input_imgH = 288
input_imgW = 512


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def GenerateMeshgrid():
    for index in range(headNum):
        for i in range(mapSize[index][0]):
            for j in range(mapSize[index][1]):
                meshgrid.append(j)
                meshgrid.append(i)


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + exp(-x))


def postprocess(out, img_h, img_w):
    print('postprocess ... ')

    detectResult = []

    output = []
    output.append(out['conv_blob44'].reshape((-1)))  # cls
    output.append(out['conv_blob46'].reshape((-1)))  # reg
    output.append(out['conv_blob47'].reshape((-1)))  # ce

    output.append(out['conv_blob50'].reshape((-1)))  # cls
    output.append(out['conv_blob52'].reshape((-1)))  # reg
    output.append(out['conv_blob53'].reshape((-1)))  # ce

    output.append(out['conv_blob56'].reshape((-1)))  # cls
    output.append(out['conv_blob58'].reshape((-1)))  # reg
    output.append(out['conv_blob59'].reshape((-1)))  # ce

    scale_h = img_h / input_imgH
    scale_w = img_w / input_imgW

    gridIndex = -2

    for index in range(headNum):
        cls = output[index * 3 + 0]
        reg = output[index * 3 + 1]
        ce = output[index * 3 + 2]

        for h in range(mapSize[index][0]):
            for w in range(mapSize[index][1]):

                gridIndex += 2

                ce_val = sigmoid(ce[h * mapSize[index][1] + w])

                for cl in range(class_num):
                    cls_val = sigmoid(cls[cl * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * ce_val;

                    if cls_val > objectThresh:
                        cx = (meshgrid[gridIndex + 0] + reg[0 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        cy = (meshgrid[gridIndex + 1] + reg[1 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        xf = exp(reg[2 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]
                        yf = exp(reg[3 * mapSize[index][0] * mapSize[index][1] + h * mapSize[index][1] + w]) * strides[index]

                        xmin = (cx - xf / 2) * scale_w
                        ymin = (cy - yf / 2) * scale_h
                        xmax = (cx + xf / 2) * scale_w
                        ymax = (cy + yf / 2) * scale_h
                        
                        xmin = xmin if xmin > 0 else 0
                        ymin = ymin if ymin > 0 else 0
                        xmax = xmax if xmax < img_w else img_w
                        ymax = ymax if ymax < img_h else img_h
                        
                        box = DetectBox(cl, cls_val, xmin, ymin, xmax, ymax)
                        detectResult.append(box)
    # NMS 过程
    print('detectResult:', len(detectResult))
    predBox = NMS(detectResult)

    return predBox


def preprocess(src):
    img = cv2.resize(src, (input_imgW, input_imgH))
    img = img.astype(np.float32)
    img = img * 0.00392156
    return img


def detect(imgfile):
    origimg = cv2.imread(imgfile)
    img_h, img_w = origimg.shape[:2]
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['blob1'].data[...] = img
    out = net.forward()
    predbox = postprocess(out, img_h, img_w)

    print(len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(origimg, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        ptext = (xmin, ymin)
        title = CLASSES[classId] + "%.2f" % score
        cv2.putText(origimg, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./test_caffe_result.jpg', origimg)
    # cv2.imshow("test", origimg)
    # cv2.waitKey(0)


if __name__ == '__main__':
    print('This is main .... ')
    GenerateMeshgrid()
    detect('./test.jpg')