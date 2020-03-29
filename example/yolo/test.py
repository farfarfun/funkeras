#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-07-19 10:29:34
#   Description :
#
# ================================================================

import os
import shutil

import cv2
import numpy as np
import tensorflow as tf

import notekeras.temp.yolo3.core.utils as utils
from notekeras.component.yolo import YoloModel
from notekeras.temp.yolo3.core.yolov3 import decode
from notekeras.utils import read_lines, image_resize, draw_bbox

INPUT_SIZE = 416

classes = read_lines("./data/classes/coco.names")
NUM_CLASS = len(classes)

predicted_dir_path = '../mAP/predicted'
ground_truth_dir_path = '../mAP/ground-truth'
if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
if os.path.exists("./data/detection/"): shutil.rmtree("./data/detection/")

os.mkdir(predicted_dir_path)
os.mkdir(ground_truth_dir_path)
os.mkdir("./data/detection/")

# Build Model
input_layer = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])


def get_anchors():
    """loads the anchors from a file"""

    anchors = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


yolo_model = YoloModel(9 // 3,
                       80,
                       inputs=input_layer,
                       as_model=True,
                       name='Yolo',
                       anchors=get_anchors(),
                       layer_depth=10)
yolo_model.load_weights("/Users/liangtaoniu/workspace/MyDiary/tmp/models/yolo/configs/yolov3.h5")
feature_maps = yolo_model.output[::-1]

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
model.load_weights("./yolov3")

with open("./data/dataset/yymnist_test.txt", 'r') as annotation_file:
    for num, line in enumerate(annotation_file):
        annotation = line.strip().split()
        image_path = annotation[0]
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

        if len(bbox_data_gt) == 0:
            bboxes_gt = []
            classes_gt = []
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

        print('=> ground truth of %s:' % image_name)
        num_bbox_gt = len(bboxes_gt)
        with open(ground_truth_path, 'w') as f:
            for i in range(num_bbox_gt):
                class_name = classes[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
        print('=> predict result of %s:' % image_name)
        predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
        # Predict Process
        image_size = image.shape[:2]
        image_data = image_resize(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        if "./data/detection/" is not None:
            image = draw_bbox(image, bboxes, classes=classes)
            cv2.imwrite("./data/detection/" + image_name, image)

        with open(predict_result_path, 'w') as f:
            for bbox in bboxes:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
