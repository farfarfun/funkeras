import os
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.python.ops import control_flow_ops

import notekeras.temp.yolo3.core.utils as utils
from notekeras.component.yolo import YoloModel
from notekeras.utils import image_resize

STRIDES = np.array([8, 16, 32])
IOU_LOSS_THRESH = 0.5


def get_anchors(index=1):
    """loads the anchors from a file"""
    if index == 0:
        anchors = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
    else:
        anchors = '1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875'
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


class YoloBody:
    def __init__(self, anchors, num_classes, num_anchors=None, root='./', *args, **kwargs):
        self.anchors = anchors
        self.num_anchors = num_anchors or len(anchors)
        self.num_classes = num_classes

        self.input_shape = (416, 416)

        self.root = root
        self.yolo_model = self.predict_model = self.train_model = None
        self.build()
        # self.debug()

    def build(self):
        image_input = Input(shape=(None, None, 3))
        yolo_model = YoloModel(self.num_anchors // 3,
                               self.num_classes,
                               inputs=image_input,
                               as_model=True,
                               name='Yolo',
                               anchors=self.anchors,
                               layer_depth=10)

        self.yolo_model = Model(yolo_model.input, yolo_model.output)

        # ###############################################
        h, w = self.input_shape

        y_true = [Input(shape=(h // 32, w // 32, self.num_anchors // 3, self.num_classes + 5)),
                  Input(shape=(h // 16, w // 16, self.num_anchors // 3, self.num_classes + 5)),
                  Input(shape=(h // 8, w // 8, self.num_anchors // 3, self.num_classes + 5))
                  ]

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments={
            'anchors': self.anchors,
            'num_classes': self.num_classes,
            'ignore_thresh': 0.5})([*self.yolo_model.output, *y_true])
        self.train_model = Model([self.yolo_model.input, *y_true], model_loss)

        # ###############################################
        feature_maps = self.yolo_model.output[::-1]

        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = self.decode(fm, index=i)
            bbox_tensors.append(bbox_tensor)

        self.predict_model = tf.keras.Model(self.yolo_model.input, bbox_tensors)
        # ###############################################
        feature_maps = self.yolo_model.output[::-1]

        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = self.decode(fm, index=i)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

        self.predict_model2 = tf.keras.Model(self.yolo_model.input, bbox_tensors)

    def decode(self, conv_output, index=0):
        anchors = np.array(self.anchors).reshape([3, 3, 2])

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + self.num_classes))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[index]
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors[index]) * STRIDES[index]
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    # 返回预测结果
    def predict_result(self, image_path=None, original_image=None):
        if original_image is None and image_path is None:
            raise Exception("不能都为None")

        if original_image is None:
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = image_resize(np.copy(original_image), self.input_shape)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        return original_image, self.predict_model.predict(image_data)

    # 返回框
    def predict_box(self, image_path=None, original_image=None):
        original_image, pred_box = self.predict_result(image_path, original_image=original_image)

        original_image_size = original_image.shape[:2]
        pred_box = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_box]
        pred_box = tf.concat(pred_box, axis=0)
        boxes = utils.postprocess_boxes(pred_box, original_image_size, self.input_shape[0], 0.3)
        boxes = utils.nms(boxes, 0.45, method='nms')

        return original_image, boxes

    def train(self):
        trainset = Dataset('train')
        for image_data, target in trainset:
            self.train_model.evaluate(image_data, target[::-1])

    def load_weights(self, filepath, freeze_body=2):
        self.yolo_model.load_weights(filepath)

        print('Load weights success {}.'.format(filepath))

        if freeze_body in [1, 2]:
            num = (20, len(self.yolo_model.layers) - 2)[freeze_body - 1]
            for i in range(num):
                self.yolo_model.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(self.yolo_model.layers)))

    def debug(self):
        plot_model(self.yolo_model,
                   to_file=self.root + 'models/yolo-body.png',
                   show_shapes=True)
        plot_model(self.yolo_model,
                   to_file=self.root + 'models/yolo-body-expand.png',
                   show_shapes=True,
                   expand_nested=True)
        plot_model(self.yolo_model,
                   to_file=self.root + 'models/yolo-train.png',
                   show_shapes=True)
        plot_model(self.yolo_model,
                   to_file=self.root + 'models/yolo-train-expand.png',
                   show_shapes=True,
                   expand_nested=True)

    def yolo_eval(self, inputs, image_shape, max_boxes=20, score_threshold=.6, iou_threshold=.5):
        """Evaluate YOLO model on given input and return filtered boxes."""
        yolo_outputs = self.yolo_model.predict(inputs)
        print(yolo_outputs[0][0][0][0][0])

        num_layers = len(yolo_outputs)

        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
        input_shape = K.shape(yolo_outputs[0])[1:3] * 32
        boxes = []
        box_scores = []
        for l in range(num_layers):
            _boxes, _box_scores = self.yolo_eval_boxes_and_scores(yolo_outputs[l],
                                                                  self.anchors[anchor_mask[l]],
                                                                  input_shape,
                                                                  image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)

        boxes, box_scores = K.concatenate(boxes, axis=0), K.concatenate(box_scores, axis=0)

        mask = box_scores >= score_threshold
        max_boxes_tensor = K.constant(max_boxes, dtype='int32')

        out_boxes, out_scores, out_classes = [], [], []

        for c in range(self.num_classes):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c
            out_boxes.append(class_boxes)
            out_scores.append(class_box_scores)
            out_classes.append(classes)

        out_boxes = K.concatenate(out_boxes, axis=0)
        out_scores = K.concatenate(out_scores, axis=0)
        out_classes = K.concatenate(out_classes, axis=0)

        return out_boxes, out_scores, out_classes

    def yolo_eval_boxes_and_scores(self, feats, anchors, input_shape, image_shape):
        """Process Conv layer output"""
        box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, self.num_classes, input_shape)
        boxes = self.yolo_eval_correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = K.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = K.reshape(box_scores, [-1, self.num_classes])
        return boxes, box_scores

    @staticmethod
    def yolo_eval_correct_boxes(box_xy, box_wh, input_shape, image_shape):
        """Get corrected boxes"""
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))
        new_shape = K.round(image_shape * K.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = K.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ])

        # Scale boxes back to original image shape.
        boxes *= K.concatenate([image_shape, image_shape])
        return boxes


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_height, grid_width = grid_shape[0], grid_shape[1]  # K.shape(feats)[1:3]  # height, width
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_width), [1, -1, 1, 1]), [grid_height, 1, 1, 1])
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_height), [-1, 1, 1, 1]), [1, grid_width, 1, 1])

    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, feats.dtype)

    feats = K.reshape(feats, [-1, grid_height, grid_width, num_anchors, num_classes + 5])
    # 0-1 网格的起点 将feats中xy的值，经过sigmoid归一化，再加上相应的grid的二元组，再除以网格边长，归一化；
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], feats.dtype)

    # 2-3 网格的宽高 将feats中wh的值，经过exp正值化，再乘以anchors_tensor的anchor box，再除以图片宽高，归一化；
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], feats.dtype)

    if calc_loss is True:
        return grid, feats, box_xy, box_wh

    # 框置信度：将feats中confidence值，经过sigmoid归一化；
    box_confidence = K.sigmoid(feats[..., 4:5])

    # 类别置信度：将feats中class_probs值，经过sigmoid归一化；
    box_class_probs = K.sigmoid(feats[..., 5:])
    return box_xy, box_wh, box_confidence, box_class_probs


def box_iou(b1, b2):
    """Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    """

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def bb_intersection_over_union(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    """Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    args Lambda层的输入，即model_body.output和y_true的组合
    anchors 二维数组，结构是(9, 2)，即9个anchor box；
    num_classes 类别数
    ignore_thresh 过滤阈值
    print_loss 打印损失函数的开关
    Returns
    -------
    loss: tensor, shape=(1,)
    :param ignore_thresh:
    :param num_classes:
    :param args:
    :param anchors:
    :param print_loss:
    """
    num_layers = len(anchors) // 3
    y_pred = args[:num_layers]
    y_true = args[num_layers:]

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = K.cast(K.shape(y_pred[0])[1:3] * 32, K.dtype(y_true[0]))

    grid_shapes = [K.cast(K.shape(y_pred[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    batch_size = K.shape(y_pred[0])[0]  # batch size, tensor
    batch_size2 = K.cast(batch_size, K.dtype(y_pred[0]))

    for l in range(num_layers):

        # y_true的第4位，即是否含有物体，含有是1，不含是0。
        true_raw_xy = y_true[l][..., :2]
        true_raw_wh = y_true[l][..., 2:4]
        true_mask_object = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(y_pred[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)

        pred_raw_xy = raw_pred[..., 0:2]
        pred_raw_wh = raw_pred[..., 2:3]
        pred_mask_object = raw_pred[..., 4:5]
        pred_class_probs = raw_pred[..., 5:]

        pred_box = K.concatenate([pred_xy, pred_wh])

        # 在网格中的中心点xy，偏移数据，值的范围是0~1；y_true的第0和1位是中心点xy的相对位置，范围是0~1；
        true_raw_xy = true_raw_xy * grid_shapes[l][::-1] - grid

        # 在网络中的wh针对于anchors的比例，再转换为log形式，范围是有正有负；y_true的第2和3位是宽高wh的相对位置，范围是0~1；
        true_raw_wh = K.log(true_raw_wh / anchors[anchor_mask[l]] * input_shape[::-1])
        true_raw_wh = K.switch(true_mask_object, true_raw_wh, K.zeros_like(true_raw_wh))  # avoid log(0)=-inf

        # 与物体框的大小有关，2减去相对面积，值得范围是(1~2)，计算wh权重，取值范围(1~2)；
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.

        object_mask_bool = K.cast(true_mask_object, 'bool')
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = control_flow_ops.while_loop(lambda b, *args: b < batch_size, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # ########计算 四个损失值，binary_crossentropy 避免过拟合###############################################
        # 1.中心点的损失值。binary_crossentropy是二值交叉熵。
        loss_xy = true_mask_object * box_loss_scale * K.binary_crossentropy(true_raw_xy, pred_raw_xy, from_logits=True)
        # loss_xy = 0
        # 2.宽高的损失值。除此之外，额外乘以系数0.5，平方K.square()。
        loss_wh = true_mask_object * box_loss_scale * 0.5 * K.square(true_raw_wh - pred_raw_wh)

        # 3.置信度损失值。两部分组成，存在物体的损失值 & 不存在物体的损失值，其中乘以忽略掩码ignore_mask，忽略预测框中IoU大于阈值的框。
        loss_conf = K.binary_crossentropy(true_mask_object, pred_mask_object, from_logits=True) * (
                true_mask_object + (1 - true_mask_object) * ignore_mask)

        # 4.类别损失值。
        loss_class = true_mask_object * K.binary_crossentropy(true_class_probs, pred_class_probs, from_logits=True)

        # 将各部分损失值的和，除以均值，累加，作为最终的图片损失值。
        loss_xy = K.sum(loss_xy) / batch_size2
        loss_wh = K.sum(loss_wh) / batch_size2
        loss_conf = K.sum(loss_conf) / batch_size2
        loss_class = K.sum(loss_class) / batch_size2

        loss += loss_xy + loss_wh + loss_conf + loss_class
        if print_loss:
            loss = tf.Print(loss, [loss, loss_xy, loss_wh, loss_conf, loss_class, K.sum(ignore_mask)], message='loss: ')
    return loss


class Dataset(object):
    """implement Dataset here"""

    def __init__(self,
                 annot_path,
                 anchors=None,
                 classes=None):
        self.annot_path = annot_path
        self.input_sizes = [416]
        self.batch_size = 4
        self.data_aug = True

        self.train_input_sizes = [416]
        self.strides = np.array([8, 16, 32])
        self.classes = classes
        self.num_classes = len(self.classes)
        self.anchors = np.array(anchors).reshape([3, 3, 2])
        self.anchor_per_scale = 3
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                        bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):

        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes = image_resize(np.copy(image), [self.train_input_size, self.train_input_size],
                                     np.copy(bboxes))
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
