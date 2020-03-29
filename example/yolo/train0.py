import colorsys
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from notekeras.model.yolo import YoloBody
from notekeras.model.yolo import yolo_loss
from notekeras.utils import get_random_data

tf.config.experimental_run_functions_eagerly(True)


def get_anchors():
    """loads the anchors from a file"""

    anchors = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size  # 原始图像是1200x1800
    w, h = size  # 转换为416x416
    scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换比例
    nw = int(iw * scale)  # 新图像的宽，保证新图像是等比下降的
    nh = int(ih * scale)  # 新图像的高

    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    new_image = Image.new('RGB', size, (128, 128, 128))  # 生成灰色图像
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    return new_image


class YoloTrain(object):
    def __init__(self):
        root = '/Users/liangtaoniu/workspace/MyDiary/tmp/models/yolo/'
        self.root = root
        self.model_path = root + 'configs/yolov3.h5'
        self.anchors_path = root + 'configs/yolo_anchors.txt'  # Anchors
        self.classes_path = root + 'configs/coco_classes.txt'  # 类别文件
        self.weights_path = root + 'configs/yolov3.h5'
        self.log_dir = root + 'logs/001/'  # 日志文件夹

        self.annotation_path = root + 'dataset/WIDER_train.txt'  # 数据

        self.model_body = self.model = None

        self.class_names = self._get_class()  # 获取类别
        self.colors = self.__get_colors(self.class_names)

        self.num_classes = len(self.class_names)
        self.input_shape = (416, 416)
        self.model_image_size = (416, 416)
        self.anchors = get_anchors()

    @staticmethod
    def __get_colors(names):
        # 不同的框，不同的颜色
        hsv_tuples = [(float(x) / len(names), 1., 1.)
                      for x in range(len(names))]  # 不同颜色
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))  # RGB
        np.random.seed(10101)
        np.random.shuffle(colors)
        np.random.seed(None)

        return colors

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding='utf8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def build(self):
        # make sure you know what you freeze
        self.create_model_yolo(load_pretrained=True, freeze_body=2, )
        # self.create_model_train()

        # self.model.summary()

    def create_model_yolo(self, load_pretrained=True, freeze_body=2, weights_path=None):
        """create the training model"""
        weights_path = weights_path or self.weights_path
        image_input = Input(shape=(None, None, 3))
        num_anchors = len(self.anchors)
        self.model_body = YoloBody(num_anchors=num_anchors,
                                   num_classes=self.num_classes,
                                   anchors=self.anchors,
                                   root=self.root,
                                   inputs=image_input,
                                   as_model=True,
                                   layer_depth=10)

        print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, self.num_classes))

        if load_pretrained:
            self.model_body.load_weights(weights_path, freeze_body)

    def create_model_train(self):
        num_anchors = len(self.anchors)
        h, w = self.input_shape

        y_true = [Input(shape=(h // 32, w // 32, num_anchors // 3, self.num_classes + 5)),
                  Input(shape=(h // 16, w // 16, num_anchors // 3, self.num_classes + 5)),
                  Input(shape=(h // 8, w // 8, num_anchors // 3, self.num_classes + 5))
                  ]

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments={
            'anchors': self.anchors,
            'num_classes': self.num_classes,
            'ignore_thresh': 0.5})([*self.model_body.output, *y_true])
        self.model = Model([self.model_body.input, *y_true], model_loss)

    def train(self):
        logging = TensorBoard(log_dir=self.log_dir)
        checkpoint = ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True,
                                     save_best_only=True, period=3)  # 只存储weights，
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)  # 当评价指标不在提升时，减少学习率
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)  # 测试集准确率，下降前终止

        val_split = 0.1  # 训练和验证的比例
        with open(self.annotation_path) as f:
            lines = f.readlines()
        np.random.seed(47)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)  # 验证集数量
        num_train = len(lines) - num_val  # 训练集数量

        """
        把目标当成一个输入，构成多输入模型，把loss写成一个层，作为最后的输出，搭建模型的时候，
        就只需要将模型的output定义为loss，而compile的时候，
        直接将loss设置为y_pred（因为模型的输出就是loss，所以y_pred就是loss），
        无视y_true，训练的时候，y_true随便扔一个符合形状的数组进去就行了。
        """
        steps = [1, 2]
        if 1 in steps:
            # 使用定制的 yolo_loss Lambda层
            self.model.compile(optimizer=Adam(lr=1e-3), loss={
                'yolo_loss': lambda y_true, y_pred: y_pred})  # 损失函数

            batch_size = 32  # batch尺寸
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            self.model.fit_generator(
                data_generator_wrapper(lines[:num_train],
                                       batch_size,
                                       self.input_shape,
                                       self.anchors,
                                       self.num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=data_generator_wrapper(lines[num_train:],
                                                       batch_size,
                                                       self.input_shape,
                                                       self.anchors,
                                                       self.num_classes),
                validation_steps=max(1, num_val // batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint])
            self.model.save_weights(self.log_dir + 'trained_weights_stage_1.h5')  # 存储最终的参数，再训练过程中，通过回调存储

        if True:  # 全部训练
            for i in range(len(self.model.layers)):
                self.model.layers[i].trainable = True

            self.model.compile(optimizer=Adam(lr=1e-4),
                               loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
            print('Unfreeze all of the layers.')

            batch_size = 16  # note that more GPU memory is required after unfreezing the body
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

            self.model.fit_generator(
                data_generator_wrapper(lines[:num_train],
                                       batch_size,
                                       self.input_shape,
                                       self.anchors,
                                       self.num_classes),
                steps_per_epoch=max(1, num_train // batch_size),
                validation_data=data_generator_wrapper(lines[num_train:],
                                                       batch_size,
                                                       self.input_shape,
                                                       self.anchors,
                                                       self.num_classes),
                validation_steps=max(1, num_val // batch_size),
                epochs=100,
                initial_epoch=50,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])
            self.model.save_weights(self.log_dir + 'trained_weights_final.h5')

    def detect_image(self, img_path):
        image = Image.open(img_path)
        start = timer()
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))  # 填充图像
        image_data = np.array(boxed_image, dtype='float32')

        print('detector size {}'.format(image_data.shape))

        image_data /= 255.  # 转换0~1
        image_data = np.expand_dims(image_data, 0)  # 添加批次维度，将图片增加1维

        image_shape = np.expand_dims(np.array([image.size[1], image.size[0]]), 0)

        # 参数盒子、得分、类别；输入图像0~1，4维；原始图像的尺寸
        out_boxes, out_scores, out_classes = self.model_body.yolo_eval(
            image_data,
            image_shape,
            score_threshold=0.13,
            max_boxes=25
        )

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  # 检测出的框

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))  # 字体
        thickness = (image.size[0] + image.size[1]) // 512  # 厚度
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]  # 类别
            box = out_boxes[i]  # 框
            score = out_scores[i]  # 执行度

            label = '{} {:.2f}'.format(predicted_class, score)  # 标签
            draw = ImageDraw.Draw(image)  # 画图
            label_size = draw.textsize(label, font)  # 标签文字

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))  # 边框

            if top - label_size[1] >= 0:  # 标签文字
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):  # 画框
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle(  # 文字背景
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)  # 文案
            del draw

        end = timer()
        print(end - start)  # 检测执行时间
        return image


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """data generator for fit_generator"""
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)  # 获取图片和盒子
            image_data.append(image)  # 添加图片
            box_data.append(box)  # 添加盒子
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)  # 真值
        yield [image_data] + y_true, np.zeros(batch_size)


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    """Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    """
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,
                        grid_shapes[l][0],
                        grid_shapes[l][1],
                        len(anchor_mask[l]),
                        5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def test_of_detect_objects_of_image():
    yolo = YoloTrain()
    yolo.build()
    img_path = '/Users/liangtaoniu/workspace/MyDiary/tmp/models/yolo/test/bb.jpg'
    r_image = yolo.detect_image(img_path)
    r_image.save(img_path.split('.')[0] + '_.png')


test_of_detect_objects_of_image()
