import os
import shutil

import numpy as np
import tensorflow as tf

from notekeras.model.yolo import Dataset
from notekeras.model.yolo import YoloBody
from notekeras.temp.yolo3.core.yolov3 import compute_loss
from notekeras.utils import read_lines

root = '/Users/liangtaoniu/workspace/MyDiary/notechats/notekeras/example/yolo'
classes = read_lines(root + "/data/classes/coco.names")
annot_path = root + "/data/dataset/yymnist_train.txt"


def get_anchors():
    anchors = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
    anchors = '1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875'
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


anchors = get_anchors()

trainset = Dataset(annot_path=annot_path, anchors=anchors, classes=classes)
logdir = "./data/log"
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = 2 * steps_per_epoch
total_steps = 30 * steps_per_epoch

input_tensor = tf.keras.layers.Input([416, 416, 3])

yolo_body = YoloBody(anchors=anchors, num_classes=len(classes))
yolo_body.load_weights("/Users/liangtaoniu/workspace/MyDiary/tmp/models/yolo/configs/yolov3.h5", freeze_body=3)

model = yolo_body.predict_model2

optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)


def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_loss(num_class=len(classes),
                                      pred=pred,
                                      conv=conv,
                                      label=target[i][0],
                                      bboxes=target[i][1],
                                      i=i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                           giou_loss, conf_loss,
                                                           prob_loss, total_loss))
        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * 1e-3
        else:
            lr = 1e-6 + 0.5 * (1e-3 - 1e-6) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


for epoch in range(30):
    for image_data, target in trainset:
        train_step(image_data, target)
    model.save_weights("./yolov3")
