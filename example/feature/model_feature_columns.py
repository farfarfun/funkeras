import random

import demjson
import numpy as np
import pandas as pd
import tensorflow as tf
from notekeras.component.transformer import EncoderComponent
from notekeras.features import ParseFeatureConfig
from sklearn.model_selection import train_test_split
from tensorflow import feature_column, keras
from tensorflow.keras import backend, layers
from tensorflow.keras.utils import plot_model

backend.set_floatx('float32')


def get_data():
    def arr_c(n):
        res = []
        fea_list = ["a", 'b', 'c', 'e', 'f']
        for i in range(0, n):
            random.shuffle(fea_list)
            res.append(np.array(fea_list[:4]))
        return res

    def arr_c2(n):
        res = []
        list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i in range(0, n):
            random.shuffle(list)
            res.append(np.array(list[:4]))
        return res

    URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
    dataframe = pd.read_csv(URL)
    dataframe['arr'] = arr_c(len(dataframe))
    dataframe['arr2'] = arr_c(len(dataframe))
    #dataframe['arr3'] = arr_c(len(dataframe))
    dataframe['thal2'] = dataframe['thal']
    dataframe = pd.DataFrame(dataframe)

    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('target')
        arr = np.vstack(dataframe.pop('arr'))
        arr2 = np.vstack(dataframe.pop('arr2'))

        data = dict(dataframe)
        data['arr'] = arr
        data['arr2'] = arr2
        ds = tf.data.Dataset.from_tensor_slices((data, labels))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    batch_size = 5
    train_d = df_to_dataset(train, batch_size=batch_size)
    val_d = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_d = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    return train_d, val_d, test_d


train_ds, val_ds, test_ds = get_data()


def compare2():
    feature_json = open('model_feature.json', 'r').read()
    feature_json = demjson.decode(feature_json)
    parse = ParseFeatureConfig()

    l0 = parse.parse_feature_json(feature_json['layer0'])
    # l01 = parse.parse_feature_json(feature_json['layer4'])

    la1, la1_length = parse.parse_sequence_feature_json(feature_json['layer3'])
    la2, la2_length = parse.parse_sequence_feature_json(feature_json['layer2'])
    # la3, la3_length = parse.parse_sequence_feature_json(feature_json['layer5'])

    la1 = EncoderComponent(name='wrap', head_num=2,
                           hidden_dim=2, layer_depth=0)(la1)

    lb1 = tf.keras.backend.mean(la1, axis=1)
    lb2 = tf.keras.backend.mean(la2, axis=1)
    # lb3 = la3

    l0 = tf.keras.backend.concatenate([l0, lb1, lb2])

    l1 = layers.Dense(128, activation='relu')(l0)
    l2 = layers.Dense(64, activation='relu')(l1)
    l3 = layers.Dense(1, activation='sigmoid')(l2)

    model = keras.models.Model(inputs=list(
        parse.feature_dict.values()), outputs=[l3])
    model.compile(optimizer='adam', loss='binary_crossentropy', )
    model.summary()
    #plot_model(model, to_file='feature2.png', show_shapes=True)
    plot_model(model, to_file='feature2.png')

    model.fit(train_ds, validation_data=val_ds, epochs=5)


compare2()
