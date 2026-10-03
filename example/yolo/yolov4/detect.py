import os

from cv2 import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from fundrive.lanzou import LanZouCloud, CodeDetail, download
import funkeras.models.yolo4.core.utils as utils
from funkeras.backend import plot_model
from funkeras.models.yolo4.core.yolov4 import YOLO, decode
from funkeras.models.yolo4.core.yolov4 import filter_boxes

# 数据/权重根目录因人而异，没有可移植的默认值，统一通过环境变量传入，
# 避免在模块导入阶段就依赖作者本机路径并发起网络下载。
data_root = os.environ.get("FUNKERAS_YOLOV4_ROOT", os.path.join(os.getcwd(), "example", "yolo") + "/")


def save_tf(
    weights=data_root + "/models/yolov4.weights",
    output_h5=data_root + "/models/yolov4-416.h5",
    input_size=416,
    score_thres=0.2,
    framework="tf",
    model_name="yolov4",
):
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(model=model_name)

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, model_name)
    bbox_tensors = []
    prob_tensors = []
    for i, fm in enumerate(feature_maps):
        if i == 0:
            output_tensors = decode(fm, input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
        elif i == 1:
            output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
        else:
            output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
        bbox_tensors.append(output_tensors[0])
        prob_tensors.append(output_tensors[1])

    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    boxes, pred_conf = filter_boxes(
        pred_bbox, pred_prob, score_threshold=score_thres, input_shape=tf.constant([input_size, input_size])
    )
    pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)

    plot_model(model, to_file="yolov4.png", show_shapes=True)

    if os.path.exists(output_h5):
        model.load_weights(output_h5)
    else:
        utils.load_weights(model, weights, model_name)
        model.save_weights(output_h5)

    model.summary()
    return model


def detect(
    image_path=data_root + "/data/images/kite.jpg",
    output=data_root + "/yolov4/result2.png",
    input_size=416,
    iou=0.45,
    score=0.025,
):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.0

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    model = save_tf()
    batch_data = tf.constant(images_data)
    pred_bbox = model(batch_data)

    boxes = pred_bbox[:, :, 0:4]
    pred_conf = pred_bbox[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score,
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(original_image, pred_bbox, classes=classes)

    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(output, image)


if __name__ == "__main__":
    # yolov4.weights
    download("https://wws.lanzous.com/b01hjn3yd", dir_pwd=data_root + "/models/")
    # yolov4-416.h5
    # download('https://wws.lanzous.com/b01hl9lej', dir_pwd=data_root + '/models/')
    classes = utils.read_class_names(data_root + "/data/classes/coco.names")
    detect()
