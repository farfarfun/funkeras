import os
import random

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

from notekeras.utils.image import image_resize, read_image_batch
from notekeras.utils.util import compose


class SimilarityDataset(object):
    def __init__(self, annotation_path,
                 tfrecord_path='./data/dataset.tfrecord',

                 repeat=5,
                 batch_size=4):
        self.batch_size = batch_size

        self.tfrecord_path = tfrecord_path
        self.annotation_path = annotation_path

        self.input_size = [416, 416]
        self.output_size = [[52, 52], [26, 26], [13, 13]]

        self.annotations = []
        self.load_annotations()

    def load_annotations(self):
        annotations = pd.read_csv(self.annotation_path)
        annotations.fillna('', inplace=True)

        self.annotations = list(annotations.values)

        random.shuffle(self.annotations)

    def build0(self):
        if not os.path.exists(self.tfrecord_path):
            pass

        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:

            for annotation in tqdm(self.annotations.values):
                label, label1, label2 = annotation[1], annotation[4], annotation[8]
                path1, path2 = annotation[2], annotation[6]
                boxs1, boxs2 = eval(annotation[3]), eval(annotation[7])
                text1, text2 = annotation[5], annotation[9]

                image1, image2 = cv2.imread(path1), cv2.imread(path2)
                image1 = image1[boxs1[1]:boxs1[3], boxs1[0]:boxs1[2], :]
                image2 = image2[boxs2[1]:boxs2[3], boxs2[0]:boxs2[2], :]

                image1, image2 = image_resize(image1, [256, 256]), image_resize(image2, [256, 256])

                features = tf.train.Features(
                    feature={
                        "label": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[label])),
                        "label1": tf.train.Feature(int64_list=tf.train.Int64List(value=[label1])),
                        "label2": tf.train.Feature(int64_list=tf.train.Int64List(value=[label2])),
                        "image1": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image1.astype(np.float32).tostring()])),
                        "image2": tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image2.astype(np.float32).tostring()])),
                    }
                )
                tf_example = tf.train.Example(features=features)
                serialized = tf_example.SerializeToString()
                writer.write(serialized)

        print('tfrecord done.{}'.format(len(self.annotations)))

        def parse_fn(example_proto):
            features = {
                "label": tf.io.FixedLenFeature((), tf.int64),
                "image1": tf.io.FixedLenFeature((), tf.string),
                "image2": tf.io.FixedLenFeature((), tf.string),
                # "label3": tf.io.FixedLenFeature((), tf.string),
            }

            parsed_features = tf.io.parse_single_example(example_proto, features)

            label = parsed_features['label']
            image1 = K.reshape(tf.io.decode_raw(parsed_features['image1'], tf.float32), [256, 256, 3])
            image2 = K.reshape(tf.io.decode_raw(parsed_features['image2'], tf.float32), [256, 256, 3])

            return (image1, image2), label

        self.dataset = tf.data.TFRecordDataset(self.tfrecord_path).map(parse_fn)

    def build(self):
        image1_list = []
        image2_list = []
        label_list = []
        batch_num = 0

        for annotation in self.annotations:
            label, label1, label2 = annotation[1], annotation[4], annotation[8]
            path1, path2 = annotation[2], annotation[6]
            boxs1, boxs2 = eval(annotation[3]), eval(annotation[7])
            text1, text2 = annotation[5], annotation[9]

            image1, image2 = cv2.imread(path1), cv2.imread(path2)
            image1 = image1[boxs1[1]:boxs1[3], boxs1[0]:boxs1[2], :]
            image2 = image2[boxs2[1]:boxs2[3], boxs2[0]:boxs2[2], :]

            image1, image2 = image_resize(image1, [256, 256]), image_resize(image2, [256, 256])
            image1_list.append(image1)
            image2_list.append(image2)
            label_list.append(label)
            batch_num += 1
            if batch_num == self.batch_size:
                batch_num = 0
                yield (np.array(image1_list), np.array(image2_list)), np.array(label_list)


class SimilarityModel:
    def __init__(self, root):
        self.root = root
        self.conv1 = self.conv2 = None
        self.flatten = self.dense = None
        self.model = self.predict_model = None

    def build(self):
        image_input1 = Input(shape=(256, 256, 3))
        image_input2 = Input(shape=(256, 256, 3))
        self.conv1 = Conv2D(64, kernel_size=3, activation='relu')
        self.conv2 = Conv2D(32, kernel_size=3, activation='relu')
        self.flatten = Flatten()
        self.dense = Dense(10, activation='softmax')

        vec1 = self.image2vector(image_input1)
        vec2 = self.image2vector(image_input2)
        output = Dense(1, activation='softmax')(Multiply()([vec1, vec2]))

        self.model = Model([image_input1, image_input2], output)
        self.predict_model = Model(image_input1, vec1)

    def image2vector(self, inputs):
        return compose(self.conv1, self.conv2, self.flatten, self.dense)(inputs)

    def debug(self):
        plot_model(self.model, to_file=self.root + 'data/models/similarity-body.png', show_shapes=True)
        plot_model(self.model, to_file=self.root + 'data/models/similarity-body-expand.png', show_shapes=True,
                   expand_nested=True)

    def train(self, dataset: SimilarityDataset, log_dir=".data/similarity/log", lr=1e-3, steps_per_epoch=None):
        logging = TensorBoard(log_dir=log_dir)

        checkpoint = ModelCheckpoint(log_dir + 'checkpoint', monitor='val_loss', )
        # 当评价指标不在提升时，减少学习率
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1)
        # 测试集准确率，下降前终止
        early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

        self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(dataset.build(),
                       steps_per_epoch=steps_per_epoch,
                       callbacks=[logging, checkpoint, reduce_lr, early_stopping],

                       )

    def predict(self, image_path, box, text=''):
        image = cv2.imread(image_path)
        image = image[box[1]:box[3], box[0]:box[2], :]
        image = image_resize(image, [256, 256])

        image = np.resize(image, [1, *np.shape(image)])

        return self.predict_model.predict(image)[0]

    def predict_batch(self, image_paths, boxs, text=''):
        image_list = []

        for i, image in enumerate(read_image_batch(image_paths)):
            box = boxs[i]
            image = image[box[1]:box[3], box[0]:box[2], :]
            image = image_resize(image, [256, 256])
            image_list.append(image)

        image_list = np.array(image_list)

        return self.predict_model.predict(image_list)
