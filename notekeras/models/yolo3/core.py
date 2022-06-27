import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm

from notekeras.component.yolo.yolo3 import YoloModel
from notekeras.utils.image import image_resize, postprocess_boxes, nms
from notekeras.utils.image import read_image_batch
from notemodel.database import load_layers

STRIDES = np.array([8, 16, 32])
IOU_LOSS_THRESH = 0.5

md5 = ['d985be8c5e5a2346dbc2ea1aedd7157d', 'aef3adffaec9270e6c4e1c2dcd13792a', '34d0f01fe6a13b2924ff363e7660ed47',
       '487c27afe4aa03e1f4d4f3e9dd8af141', '1c9bfbbd3bfcd4c2c3b5dc9216c54fb4', '2da0263ec5f7a8cdffb22093240a5f1e',
       '0c9aa2fe2d0e2f0cd8fcdf1dfbf899e8', 'b69aaaee90f4bf7af462d2c72ab037a0', '75eee2f59d59e3fe80fa66fe81b5de09',
       '1809e805472973bcb38917157d0d8066', '31ed52b46ea7034d0915cd87246ce4be', '25d6acdbd3992dd0726d996562ffce55',
       '352d1afbf4bd2a27e65f24516a5d2955', 'c1842d90922cfaf334292e7e88e18ff5', '5154d37a11a5d4fba9dd155e1486a77f',
       'f61247f4035d16283f306a61b6e427e2', '1f2a150ff854006dd2ff6b73fc941f65', '51fab7de4bf86295ad4e659c44da30e4',
       'dd2bc86e77932eee51ba5ae9b62e2fe7', '59d8e03589407bc8bc71f6eb8f87d705', 'a6df34e012c40fd22d0a5e61cc0987f4',
       '2d648332240161e4b0ee550d859a9b4e', 'd35cb84758d3bda913c201b9c1a7fc96', 'fbf9b6be356470d6987bda32c17d5acb',
       'd2f9a90186774f60f2a1a31d15013e1c', '17879c874191128d23fa2303a836756c', 'f9298feaac0997f76d73aed18b66fe89',
       '166b4c0a5545407d43f34e9a206ad9c0', 'bfa8bf83450fcbe22750d72fbab6dcdc', '2845b3ab4289c8523c023eff5c9cb144',
       'b2bc30026542fd52a7e262638046f08d', '1ac7a9a71da94c863ae129438ee72e3f', 'eb046b7de62c651698264c9a7c576aee',
       '60e85877af120063d8a519493661de6a', '366e04618334cf49c700a55701374af3', '768030d67aa8b7efbead4d9b537edbe8',
       '9285194fb9b3ead92b14da2fb9620ca6', '504817e75243020564c8a9bb448f7088', 'e21e53d58f30da29b574d27a360e89c0',
       'ae309c7ecefd3f856d7ecd1a040bdec7', '9a1508d819e698c7647f04d6d14c769e', 'f4a8c57176b974e3fc8a60977ebef166',
       '7fe247732b48dad31717a9658fb135bb', '9383fabf508bf90b2d6bd802af22e1c1', '03ae44cbe8f5a547f56c4a1e986f8bbf',
       'f96971e009d86a99201797333a92391a', '6848b99d533859b3650c0fb9fb481335', 'b8cb8f647b7afd50f225cf2ca726c458',
       'ba5a1ed672a2aa05a6318e9c5d2906a2', '4dc4c8992aab80d6c4eac10e0ad5b597', 'af6b409ca12b4d936438665b40c8b0c5',
       'd8360f69f8ea8d3a6a0ac2bd9ca1517a', 'c84b24ebd0c3192c50c447c15b483d4a', '93b1f4f605af8774cc76f1af9caabbf5',
       '45f9e3cbad812612a5aaecaec75d9a0f', '14678aa53b7a160769db818cecacb7b4', '879c0bebeac9e82e7b1340fd8f222377',
       'aab301f841911805708e4ff5e15612d4', '749a3601857dab21af4614143aa50ab4', 'c1dc231210cf38300455ddf10df028f3',
       'cc9b7a4c5640305f77e1795e55d3cb1b', 'd81ad529c962a9a089b64e7db9ca2147', 'ef600e41bc8e53e2298d313e348f1995',
       'dffaf97ae3465f8919eaefedfed40c32', 'fd4be1c5e83752fa71ba48f7afdae528', 'aca0215a4f6d84fef1105c7f8b3942f0',
       '2810294305520edfc96a8a5c3298dbc1', '359e0f4b8252af017931e690a1f4f14e', '5027af46012b18dd3004232edebf9dd0',
       '7bf9549e8f7c845636826edfda44490d', '60662f2603a92a52f3133b279ebcee33', '8f498ad9ea93fe0d3cd3e6c15f8c9208',
       '52c9166c4d112dae6afaa30f6ae2b6d8', 'd79e4fc025e8dbb1ac506d5b982972fd', '2b909ca2713f3f91682f47d3171f0e91',
       'b476a5da5664a19bd4c7e767bcdca186', '2baafa1143a0b68c3c3b8e4c6a8d4221', '07ec13e77effde4be96f0faefed653b2',
       'cf5ecadbccae701eda1ab453c8e11b81', '194cfe94f2bac2b2c6fe95b55f9c62bd', '3a00b9644883d0c6e26304aff60bf8b3',
       '1a2fa074af5690c8695234587efb4664', '1a31196cf597c4c5f24156bb5ac134ee', '0ea91048e9a14e5c1c6a394e303ce05c',
       'fd860136b9f9e485590df54147b71dc3', '76b35f540733953449c1cfef60d193c6', 'a54642e5b1c7580ce82abab8ad2f0a32',
       '118eaab119749a5b85c58dc6f0f5e9ed', '3c20adc601f32ea9de6c5662fdf74b18', '7810b87228f67addd4ba1820ff713f2f',
       'e4f2028a62d467c2e2decf0adf63cbe6', '0a565e22ee3d6ead99455707822863e2', '2d36413d6070bfbf03d36b10e83c732f',
       'ed9add4009ca7c0b2d43d82d14eac554', '4dd7db616c881cec92c79870b30baad9', '6ceed128665dab6117dcb062260d7e81',
       '6eccf4ce30ac40741452d970e17e0fd9', 'd691f46307d9c1d23bb0b48052e3b7cf', 'e939b3ea3a9d8c812b6be35c159af1ba',
       'f6ec2b4a2a3476b32130458cb697e5af', '57228c16908a6a5ee84e11f8955e22e7', '7395ea7977bc8d76d1547b1897abb751',
       'e2ea74f04b978dedd83e93d12a9f0230', '65e9aa3e7da9b272d846455cbba8db3f', 'f68261dd498bef5a6591a4b47f522c64',
       '8554a1500ca3cdbd47345d38b99628d7', 'e17255b097ef95170cb42a1ed994530f', '8a33695f4ee651c0af6a1e880818327d',
       'ceecd9822245d5a9d4f0cb18cbabd129', '96c9ef3ecb129febbf500afa1823e543', '7bb434d44243b295a81b3e08821152ac',
       '7efadafc7ba19b9688ec12bb60c5cdfe', '31e04d148375aacda73ddedaadd8a203', 'a8f29119c65f75adf820466543dd5c67',
       'd7ec0a93875785fdbb5b56e9ce9d4b14', '34d47151e7ad2d15fabb636e4fb88228', '00a81219149cb3edf7f8df9b294b0236',
       '45f5bb5c8cbb07fe729c8bceb616a598', 'ab5bebf5158a14c1a4e2841152a87645', 'ff4a4d61911b80577ccac02b28578516',
       'b0c17539cee51b3ef392c1d1aba0d003', '7f94d2597839a7003a485b3db564a8e1', 'b1731075217a0b8a983eb95a2e9af7bd',
       'e2104b4437c2d2d9137fcd782d08cd96', '272179c4bc1a469494db80b09a2c314c', '0f1ff88630b2dc0facd694a6d04e6f30',
       'bdc2b22959b1c98e2de4111b56300093', '2f516f1da9da275a83bbd35f62521906', '4d48f19ebbaaf2b5c39828a1b3bc787c',
       '0218193f2e16174d46282b325b10f6cc', '443aae900cc3988619a1b4920d8e2cb9', '2773205737363ae9b63d956a58e86fb8',
       '7a883cea4099cc12c50820662745ee60', 'c239895083753827153e8e134ebcc9c5', '481109e15006a5a4cdb53f108d71732a',
       '68995bd3694a7b1ba86a0f984fa0ec09', '41012a0d2a4db3f46d2769dc82d19ee4', '9944f7ac1572ef8a8a9af1a1b2ac096a',
       '8a626b27f3ab3350c0def6c99fdc819d', '56420dae989b9dc718e7da77f6992833', '16dc735db8f36d1288c785577546bb6f',
       'f01a13cc006a25bc76a292d9708f0e6c', '154e11d12f5fea444ede98aced8be4b9', '4c834ffb76b50a9329b3cf86bb4de5b0',
       'e3dc8ded3f1cdbb6a8de22781d97174f', 'bae4b69f3804afe1cc4b7ef88a6344be', '39373d699a052aab6ebf9b9e5483f64f']


class Dataset(object):
    def __init__(self, annotation_path, tfrecord_path='./data/dataset.tfrecord', anchors=None, classes=None, repeat=5,
                 batch_size=4):
        self.anchors = anchors
        self.classes = classes
        self.batch_size = batch_size

        self.tfrecord_path = tfrecord_path
        self.annotation_path = annotation_path

        self.input_size = [416, 416]
        self.output_size = [[52, 52], [26, 26], [13, 13]]

        self.strides = np.array([8, 16, 32])

        self.num_classes = len(self.classes)
        self.num_anchors = len(self.anchors)
        self.anchor_per_scale = 3

        self.annotations = []
        self.load_annotations()
        self.num_samples = len(self.annotations)

        self.steps_total = self.num_samples * repeat / batch_size

        self.dataset = None

        self.build()
        self.dataset_iterator = self.dataset.batch(batch_size).repeat(repeat)  # .make_one_shot_iterator()

    def load_annotations(self):
        with open(self.annotation_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        self.annotations = annotations

    def build(self):
        if not os.path.exists(self.tfrecord_path):
            pass
        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            # for annotation in self.annotations:
            for annotation in tqdm(self.annotations):
                image, label_box1, label_box2, label_box3 = self.parse_annotations(annotation)

                features = tf.train.Features(
                    feature={
                        "image": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image.astype(np.float32).tostring()])),
                        "label1": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[label_box1.astype(np.float32).tostring()])),
                        "label2": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[label_box2.astype(np.float32).tostring()])),
                        "label3": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[label_box3.astype(np.float32).tostring()])),
                    }
                )
                tf_example = tf.train.Example(features=features)
                serialized = tf_example.SerializeToString()
                writer.write(serialized)
        print('tfrecord done.{}'.format(len(self.annotations)))

        def parse_fn(example_proto):
            features = {
                "image": tf.io.FixedLenFeature((), tf.string),
                "label1": tf.io.FixedLenFeature((), tf.string),
                "label2": tf.io.FixedLenFeature((), tf.string),
                "label3": tf.io.FixedLenFeature((), tf.string),
            }
            parsed_features = tf.io.parse_single_example(example_proto, features)
            temp_size = [self.anchor_per_scale, 5 + self.num_classes]
            image = K.reshape(tf.io.decode_raw(parsed_features['image'], tf.float32), [*self.input_size, 3])
            label1 = K.reshape(tf.io.decode_raw(parsed_features['label1'], tf.float32),
                               [*self.output_size[2], *temp_size])
            label2 = K.reshape(tf.io.decode_raw(parsed_features['label2'], tf.float32),
                               [*self.output_size[1], *temp_size])
            label3 = K.reshape(tf.io.decode_raw(parsed_features['label3'], tf.float32),
                               [*self.output_size[0], *temp_size])

            return (image, label1, label2, label3), np.zeros(image.shape[0])

        self.dataset = tf.data.TFRecordDataset(self.tfrecord_path).map(parse_fn)

    def parse_annotations(self, annotation_line):
        image_data = []
        box_data = []
        image, box = self.get_random_data(annotation_line, random=True)
        image_data.append(image)
        box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = self.preprocess_true_boxes(box_data)
        return image_data, y_true[0], y_true[1], y_true[2]

    def preprocess_true_boxes(self, true_boxes):
        input_shape = self.input_size
        assert (true_boxes[..., 4] < self.num_classes).all(), 'class id must be less than num_classes'
        num_layers = len(self.anchors) // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        m = true_boxes.shape[0]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes),
                           dtype='float32') for l in range(num_layers)]

        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(self.anchors, 0)
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

    def get_random_data(self, annotation_line, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5,
                        proc_img=True):

        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = self.input_size
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(float(w) / float(iw), float(h) / float(ih))
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image_data = 0
            if proc_img:
                image = image.resize((nw, nh), Image.BICUBIC)
                new_image = Image.new('RGB', (w, h), (128, 128, 128))
                new_image.paste(image, (dx, dy))
                image_data = np.array(new_image) / 255.

            # correct boxes
            box_data = np.zeros((max_boxes, 5))
            if len(box) > 0:
                np.random.shuffle(box)
                if len(box) > max_boxes: box = box[:max_boxes]  # 最多只取20个
                box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
                box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
                box_data[:len(box)] = box

            return image_data, box_data

        def rand(a=0., b=1.):
            return np.random.rand() * (b - a) + a

        def hsv_to_rgb(hsv):
            """
            Convert hsv values to rgb.

            Parameters
            ----------
            hsv : (..., 3) array-like
               All values assumed to be in range [0, 1]

            Returns
            -------
            rgb : (..., 3) ndarray
               Colors converted to RGB values in range [0, 1]
            """
            hsv = np.asarray(hsv)

            # check length of the last dimension, should be _some_ sort of rgb
            if hsv.shape[-1] != 3:
                raise ValueError("Last dimension of input array must be 3; "
                                 "shape {shp} was found.".format(shp=hsv.shape))

            in_shape = hsv.shape
            hsv = np.array(
                hsv, copy=False,
                dtype=np.promote_types(hsv.dtype, np.float32),  # Don't work on ints.
                ndmin=2,  # In case input was 1D.
            )

            h = hsv[..., 0]
            s = hsv[..., 1]
            v = hsv[..., 2]

            r = np.empty_like(h)
            g = np.empty_like(h)
            b = np.empty_like(h)

            i = (h * 6.0).astype(int)
            f = (h * 6.0) - i
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))

            idx = i % 6 == 0
            r[idx] = v[idx]
            g[idx] = t[idx]
            b[idx] = p[idx]

            idx = i == 1
            r[idx] = q[idx]
            g[idx] = v[idx]
            b[idx] = p[idx]

            idx = i == 2
            r[idx] = p[idx]
            g[idx] = v[idx]
            b[idx] = t[idx]

            idx = i == 3
            r[idx] = p[idx]
            g[idx] = q[idx]
            b[idx] = v[idx]

            idx = i == 4
            r[idx] = t[idx]
            g[idx] = p[idx]
            b[idx] = v[idx]

            idx = i == 5
            r[idx] = v[idx]
            g[idx] = p[idx]
            b[idx] = q[idx]

            idx = s == 0
            r[idx] = v[idx]
            g[idx] = v[idx]
            b[idx] = v[idx]

            rgb = np.stack([r, g, b], axis=-1)

            return rgb.reshape(in_shape)

        def rgb_to_hsv(arr):
            """
            Convert float rgb values (in the range [0, 1]), in a numpy array to hsv
            values.

            Parameters
            ----------
            arr : (..., 3) array-like
               All values must be in the range [0, 1]

            Returns
            -------
            hsv : (..., 3) ndarray
               Colors converted to hsv values in range [0, 1]
            """
            arr = np.asarray(arr)

            # check length of the last dimension, should be _some_ sort of rgb
            if arr.shape[-1] != 3:
                raise ValueError("Last dimension of input array must be 3; "
                                 "shape {} was found.".format(arr.shape))

            in_shape = arr.shape
            arr = np.array(
                arr, copy=False,
                dtype=np.promote_types(arr.dtype, np.float32),  # Don't work on ints.
                ndmin=2,  # In case input was 1D.
            )
            out = np.zeros_like(arr)
            arr_max = arr.max(-1)
            ipos = arr_max > 0
            delta = arr.ptp(-1)
            s = np.zeros_like(delta)
            s[ipos] = delta[ipos] / arr_max[ipos]
            ipos = delta > 0
            # red is max
            idx = (arr[..., 0] == arr_max) & ipos
            out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
            # green is max
            idx = (arr[..., 1] == arr_max) & ipos
            out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
            # blue is max
            idx = (arr[..., 2] == arr_max) & ipos
            out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

            out[..., 0] = (out[..., 0] / 6.0) % 1.0
            out[..., 1] = s
            out[..., 2] = arr_max

            return out.reshape(in_shape)

        # resize image
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(.25, 2.)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data


class YoloDataset(object):
    def __init__(self, annotation_path,
                 tfrecord_path='./data/dataset.tfrecord',
                 anchors=None,
                 classes=None,
                 repeat=5,
                 batch_size=4):
        self.anchors = anchors
        self.classes = classes
        self.batch_size = batch_size

        self.tfrecord_path = tfrecord_path
        self.annotation_path = annotation_path

        self.input_size = [416, 416]
        self.output_size = [[52, 52], [26, 26], [13, 13]]

        self.strides = np.array([8, 16, 32])

        self.num_classes = len(self.classes)
        self.num_anchors = len(self.anchors)
        self.anchor_per_scale = 3

        self.annotations = []
        self.load_annotations()
        self.num_samples = len(self.annotations)

        self.steps_total = self.num_samples * repeat / batch_size

    def load_annotations(self):
        with open(self.annotation_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        self.annotations = annotations

    def build(self):
        annotations = self.annotations
        np.random.shuffle(annotations)
        for annotation in annotations:
            image, label_box1, label_box2, label_box3 = self.parse_annotations(annotation)

            inputs = (image, label_box1, label_box2, label_box3)
            outputs = (np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1))

            yield inputs, outputs

    def parse_annotations(self, annotation_line):
        image_data = []
        box_data = []
        image, box = self.get_random_data(annotation_line, random=True)
        image_data.append(image)
        box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = self.preprocess_true_boxes(box_data)
        return image_data, y_true[0], y_true[1], y_true[2]

    def preprocess_true_boxes(self, true_boxes):
        input_shape = self.input_size
        assert (true_boxes[..., 4] < self.num_classes).all(), 'class id must be less than num_classes'
        num_layers = len(self.anchors) // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        m = true_boxes.shape[0]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes),
                           dtype='float32') for l in range(num_layers)]

        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(self.anchors, 0)
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

    def get_random_data(self, annotation_line, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5,
                        proc_img=True):

        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = self.input_size
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(float(w) / float(iw), float(h) / float(ih))
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image_data = 0
            if proc_img:
                image = image.resize((nw, nh), Image.BICUBIC)
                new_image = Image.new('RGB', (w, h), (128, 128, 128))
                new_image.paste(image, (dx, dy))
                image_data = np.array(new_image) / 255.

            # correct boxes
            box_data = np.zeros((max_boxes, 5))
            if len(box) > 0:
                np.random.shuffle(box)
                if len(box) > max_boxes: box = box[:max_boxes]  # 最多只取20个
                box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
                box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
                box_data[:len(box)] = box

            return image_data, box_data

        def rand(a=0., b=1.):
            return np.random.rand() * (b - a) + a

        def hsv_to_rgb(hsv):
            """
            Convert hsv values to rgb.

            Parameters
            ----------
            hsv : (..., 3) array-like
               All values assumed to be in range [0, 1]

            Returns
            -------
            rgb : (..., 3) ndarray
               Colors converted to RGB values in range [0, 1]
            """
            hsv = np.asarray(hsv)

            # check length of the last dimension, should be _some_ sort of rgb
            if hsv.shape[-1] != 3:
                raise ValueError("Last dimension of input array must be 3; "
                                 "shape {shp} was found.".format(shp=hsv.shape))

            in_shape = hsv.shape
            hsv = np.array(
                hsv, copy=False,
                dtype=np.promote_types(hsv.dtype, np.float32),  # Don't work on ints.
                ndmin=2,  # In case input was 1D.
            )

            h = hsv[..., 0]
            s = hsv[..., 1]
            v = hsv[..., 2]

            r = np.empty_like(h)
            g = np.empty_like(h)
            b = np.empty_like(h)

            i = (h * 6.0).astype(int)
            f = (h * 6.0) - i
            p = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))

            idx = i % 6 == 0
            r[idx] = v[idx]
            g[idx] = t[idx]
            b[idx] = p[idx]

            idx = i == 1
            r[idx] = q[idx]
            g[idx] = v[idx]
            b[idx] = p[idx]

            idx = i == 2
            r[idx] = p[idx]
            g[idx] = v[idx]
            b[idx] = t[idx]

            idx = i == 3
            r[idx] = p[idx]
            g[idx] = q[idx]
            b[idx] = v[idx]

            idx = i == 4
            r[idx] = t[idx]
            g[idx] = p[idx]
            b[idx] = v[idx]

            idx = i == 5
            r[idx] = v[idx]
            g[idx] = p[idx]
            b[idx] = q[idx]

            idx = s == 0
            r[idx] = v[idx]
            g[idx] = v[idx]
            b[idx] = v[idx]

            rgb = np.stack([r, g, b], axis=-1)

            return rgb.reshape(in_shape)

        def rgb_to_hsv(arr):
            """
            Convert float rgb values (in the range [0, 1]), in a numpy array to hsv
            values.

            Parameters
            ----------
            arr : (..., 3) array-like
               All values must be in the range [0, 1]

            Returns
            -------
            hsv : (..., 3) ndarray
               Colors converted to hsv values in range [0, 1]
            """
            arr = np.asarray(arr)

            # check length of the last dimension, should be _some_ sort of rgb
            if arr.shape[-1] != 3:
                raise ValueError("Last dimension of input array must be 3; "
                                 "shape {} was found.".format(arr.shape))

            in_shape = arr.shape
            arr = np.array(
                arr, copy=False,
                dtype=np.promote_types(arr.dtype, np.float32),  # Don't work on ints.
                ndmin=2,  # In case input was 1D.
            )
            out = np.zeros_like(arr)
            arr_max = arr.max(-1)
            ipos = arr_max > 0
            delta = arr.ptp(-1)
            s = np.zeros_like(delta)
            s[ipos] = delta[ipos] / arr_max[ipos]
            ipos = delta > 0
            # red is max
            idx = (arr[..., 0] == arr_max) & ipos
            out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
            # green is max
            idx = (arr[..., 1] == arr_max) & ipos
            out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
            # blue is max
            idx = (arr[..., 2] == arr_max) & ipos
            out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

            out[..., 0] = (out[..., 0] / 6.0) % 1.0
            out[..., 1] = s
            out[..., 2] = arr_max

            return out.reshape(in_shape)

        # resize image
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(.25, 2.)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data


class YoloBody:
    def __init__(self, anchors, num_classes, num_anchors=None, root='./', *args, **kwargs):
        self.anchors = anchors
        self.num_anchors = num_anchors or len(anchors)
        self.num_classes = num_classes

        self.input_shape = (416, 416)

        self.root = root
        self.yolo_model = self.train_model = None
        self.build()

    def build(self):
        image_input = Input(shape=(None, None, 3))
        yolo_model = YoloModel(self.num_anchors // 3, self.num_classes, inputs=image_input, as_model=True, name='Yolo',
                               anchors=self.anchors, layer_depth=10)

        self.yolo_model = Model(yolo_model.input, yolo_model.output)

        # ###############################################
        h, w = self.input_shape

        y_true = [Input(shape=(h // 32, w // 32, self.num_anchors // 3, self.num_classes + 5)),
                  Input(shape=(h // 16, w // 16, self.num_anchors // 3, self.num_classes + 5)),
                  Input(shape=(h // 8, w // 8, self.num_anchors // 3, self.num_classes + 5))]

        model_loss = Lambda(yolo_loss, output_shape=(4,),
                            arguments={'anchors': self.anchors, 'num_classes': self.num_classes, 'ignore_thresh': 0.5}
                            )([*yolo_model.output, *y_true])
        loss1 = Lambda(lambda x: x[0,], output_shape=(1,), name='xy')(model_loss)
        loss2 = Lambda(lambda x: x[1,], output_shape=(1,), name='wh')(model_loss)
        loss3 = Lambda(lambda x: x[2,], output_shape=(1,), name='conf')(model_loss)
        loss4 = Lambda(lambda x: x[3,], output_shape=(1,), name='class')(model_loss)
        self.train_model = Model(inputs=[yolo_model.input, *y_true], outputs=[loss1, loss2, loss3, loss4])
        # ###############################################

    def train(self, dataset: Dataset, log_dir=".data/yolo/log", lr=1e-3, epochs=1, steps_per_epoch=None):
        logging = TensorBoard(log_dir=log_dir)

        checkpoint = ModelCheckpoint(
            log_dir + '/checkpoint',
            monitor='val_loss',
            # save_freq=2,
        )
        # 当评价指标不在提升时，减少学习率
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1)
        # 测试集准确率，下降前终止
        early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

        self.train_model.compile(optimizer=Adam(lr=lr), loss={'loss2': lambda y_true0, y_pred: y_pred})

        self.train_model.fit(dataset.dataset_iterator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                             callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    def train_iterator(self, dataset: YoloDataset, lr=1e-3, steps_per_epoch=None, epochs=1):
        # 当评价指标不在提升时，减少学习率
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, verbose=1, min_lr=0.00001)
        # 测试集准确率，下降前终止
        early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

        self.train_model.compile(optimizer=optimizers.Adadelta(lr=lr),
                                 loss={'xy': lambda y_true0, y_pred: y_pred,
                                       'wh': lambda y_true0, y_pred: y_pred,
                                       'conf': lambda y_true0, y_pred: y_pred,
                                       'class': lambda y_true0, y_pred: y_pred},
                                 loss_weights={'xy': 1., 'wh': 1., 'conf': 1., 'class': 1.})

        self.train_model.fit(dataset.build(), epochs=epochs, steps_per_epoch=steps_per_epoch,
                             callbacks=[reduce_lr, early_stopping])

    def decodes(self, outputs):
        res = []
        for i, out in enumerate(outputs):
            res.append(self.decode(out, index=i))
        return res

    def release(self):
        del self.yolo_model
        del self.train_model

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
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors[index])
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    # 返回预测结果
    def predict_result(self, image_path=None, original_image=None):
        if original_image is None:
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = image_resize(np.copy(original_image), self.input_shape)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        return original_image, self.decodes(self.yolo_model.predict(image_data)[::-1])

    # 返回框
    def predict_box(self, image_path=None, original_image=None):
        original_image, pred_box = self.predict_result(image_path, original_image=original_image)

        original_image_size = original_image.shape[:2]
        pred_box = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_box]
        pred_box = tf.concat(pred_box, axis=0)
        boxes = postprocess_boxes(pred_box, original_image_size, self.input_shape[0], 0.3)
        boxes = nms(boxes, 0.45, method='nms')

        return original_image, boxes

    def predict_result_batch(self, image_paths):
        original_image = []

        images = read_image_batch(image_paths)

        for image in images:
            image_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_data = image_resize(np.copy(image_data), self.input_shape)
            image_data = image_data.astype(np.float32)

            original_image.append(image_data)

        output = self.yolo_model.predict(np.array(original_image))
        pred_boxs = self.decodes(output[::-1])

        result_boxs = []
        for i, image in enumerate(original_image):
            pred_box = [pred_boxs[0][i], pred_boxs[1][i], pred_boxs[2][i]]

            original_image_size = images[i].shape[:2]
            pred_box = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_box]
            pred_box = tf.concat(pred_box, axis=0)
            boxes = postprocess_boxes(pred_box, original_image_size, self.input_shape[0], 0.3)
            boxes = nms(boxes, 0.45, method='nms')
            result_boxs.append(boxes)
        return result_boxs

    def load_layer_weights(self, freeze_body=2):
        print("load weight")
        load_layers(self.yolo_model.layers, model_name='yolov3', md5_list=md5)
        print("load weight done")

    def load_weights(self, filepath, freeze_body=2):

        self.yolo_model.load_weights(filepath)

        print('Load weights success {}.'.format(filepath))

        if freeze_body in [1, 2]:
            num = (20, len(self.yolo_model.layers) - 2)[freeze_body - 1]
            for i in range(num):
                self.yolo_model.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(self.yolo_model.layers)))

    def debug(self):
        plot_model(self.yolo_model, to_file=self.root + 'models/yolo-body.png', show_shapes=True)
        plot_model(self.yolo_model, to_file=self.root + 'models/yolo-body-expand.png', show_shapes=True,
                   expand_nested=True)
        plot_model(self.train_model, to_file=self.root + 'models/yolo-train.png', show_shapes=True)
        plot_model(self.train_model, to_file=self.root + 'models/yolo-train-expand.png', show_shapes=True,
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
    loss_all = [0, 0, 0, 0]

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

        # loss += loss_xy + loss_wh + loss_conf + loss_class
        loss_all[0] += loss_xy
        loss_all[1] += loss_wh
        loss_all[2] += loss_conf
        loss_all[3] += loss_class

    res = Lambda(lambda x: K.stack(x))(loss_all)

    # return loss
    return res
