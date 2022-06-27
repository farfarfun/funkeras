import numpy as np
import tensorflow as tf

from notekeras.model.yolo3 import Dataset, YoloDataset
from notekeras.model.yolo3 import YoloBody
from notekeras.utils import read_lines

root = '/Users/liangtaoniu/workspace/MyDiary/notechats/notekeras/example/yolo'
classes = read_lines(root + "/data/classes/coco.names")
annotation_path = root + "/data/dataset/yymnist_train.txt"
tf.config.experimental_run_functions_eagerly(True)


def get_anchors():
    # anchor = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
    anchor = '1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875'
    anchor = [float(x) for x in anchor.split(',')]
    return np.array(anchor).reshape(-1, 2)


anchors = get_anchors()
yolo_body = YoloBody(anchors=anchors, num_classes=len(classes))
yolo_body.debug()
yolo_body.load_weights("/Users/liangtaoniu/workspace/MyDiary/tmp/models/yolo/configs/yolov3.h5", freeze_body=3)

train_set2 = YoloDataset(annotation_path=annotation_path, anchors=anchors, classes=classes, batch_size=4)
for item in train_set2.dataset_iterator:
    print(len(item))

a = b

train_set = Dataset(annotation_path=annotation_path, anchors=anchors, classes=classes, batch_size=4)
log_dir = "/Users/liangtaoniu/workspace/MyDiary/tmp/models/yolo/log"

yolo_body.train(dataset=train_set, log_dir=log_dir)

model = yolo_body.train_model
logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
# 只存储weights，
checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                                monitor='val_loss', save_weights_only=True,
                                                save_best_only=True, period=3)
# 当评价指标不在提升时，减少学习率
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                                 verbose=1)
# 测试集准确率，下降前终止
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                  verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss={'yolo_loss': lambda y_true0, y_pred: y_pred})

model.fit_generator(train_set.dataset_iterator,
                    steps_per_epoch=train_set.steps_total,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping])
