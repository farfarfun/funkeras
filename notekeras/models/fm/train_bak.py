import datetime
import os
import pickle
import warnings

import numpy as np
import tensorflow as tf
from notedata.dataset.datas import CriteoData, ElectronicsData
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam

from notekeras.models.fm.model import *

from .model import *

criteo = CriteoData()
electronics = ElectronicsData()


def download(mode=1):
    criteo.download(mode=mode)
    electronics.download()


def train_fm(mode=1):
    # ========================= Hyper Parameters =======================
    read_part = True
    sample_num = 100000
    test_size = 0.2

    k = 10

    learning_rate = 0.001
    batch_size = 512
    epochs = 5

    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = FM(feature_columns=feature_columns, k=k)
    # model.summary()
    # ============================model checkpoint======================
    # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[checkpoint],
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_afm(mode=1):
    # ========================= Hyper Parameters =======================
    read_part = True
    sample_num = 100000
    test_size = 0.2

    embed_dim = 8

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    mode = 'avg'  # 'max', 'avg'
    model = AFM(feature_columns, mode)
    # model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/afm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint,
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_ffm(mode=1):
    # ========================= Hyper Parameters =======================
    read_part = True
    sample_num = 100000
    test_size = 0.2

    k = 8

    learning_rate = 0.001
    batch_size = 512
    epochs = 5

    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = FFM(feature_columns=feature_columns, k=k)
    # model.summary()
    # ============================model checkpoint======================
    # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[checkpoint],
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_nfm(mode=1):
    # ========================= Hyper Parameters =======================
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = NFM(feature_columns, hidden_units, dnn_dropout=dnn_dropout)
    # model.summary()

    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_deep_fm(mode=1):
    # ========================= Hyper Parameters =======================
    # you can modify your file path

    read_part = True
    sample_num = 5000000
    test_size = 0.2

    embed_dim = 8
    k = 10
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = DeepFM(feature_columns, k=k,
                   hidden_units=hidden_units, dnn_dropout=dnn_dropout)
    # model.summary()
    # ============================model checkpoint======================
    # check_path = '../save/deepfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(
            monitor='val_loss', patience=1, restore_best_weights=True)],  # checkpoint,
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_x_deep_fm(mode=1):
    # ========================= Hyper Parameters =======================

    read_part = True
    sample_num = 500000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]
    cin_size = [128, 128]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = xDeepFM(feature_columns, hidden_units, cin_size)
    # model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/xdeepfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_mf(mode=1):
    # ========================= Hyper Parameters =======================
    test_size = 0.2

    latent_dim = 32
    # implicit dataset
    implicit = False
    # use bias
    use_bias = True

    learning_rate = 0.001
    batch_size = 512
    epochs = 10

    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = MF(feature_columns, implicit, use_bias)
    # model.summary()
    # ============================model checkpoint======================
    # check_path = '../save/mf_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)
    # ============================Compile============================
    if implicit:
        model.compile(loss=binary_crossentropy, optimizer=Adam(
            learning_rate=learning_rate), metrics=[AUC()])
    else:
        model.compile(loss='mse', optimizer=Adam(
            learning_rate=learning_rate), metrics=['mse'])
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[checkpoint],
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test rmse: %f' % np.sqrt(model.evaluate(test_X, test_y)[1]))


def train_wide_deep(mode=1):
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    file = '../dataset/Criteo/train.txt'
    read_part = True
    sample_num = 5000000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = WideDeep(feature_columns, hidden_units=hidden_units,
                     dnn_dropout=dnn_dropout)
    # model.summary()
    # ============================model checkpoint======================
    # check_path = '../save/wide_deep_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ============================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ==============================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(
            monitor='val_loss', patience=1, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_dcn(mode=1):
    # ========================= Hyper Parameters =======================
    # file = '../dataset/Criteo/demo.txt'
    read_part = True
    sample_num = 5000000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = DCN(feature_columns, hidden_units, dnn_dropout=dnn_dropout)
    # model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/dcn_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(
            monitor='val_auc', patience=2, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_pnn(mode=1):
    # ========================= Hyper Parameters =======================
    file = '../dataset/Criteo/train.txt'
    read_part = True
    sample_num = 5000000
    test_size = 0.2

    embed_dim = 8

    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    mode = 'in'
    model = PNN(feature_columns, hidden_units, dnn_dropout)
    # model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/pnn_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def train_deep_cross(mode=1):
    # ========================= Hyper Parameters =======================

    read_part = True
    sample_num = 5000000
    test_size = 0.2

    embed_dim = 8
    dnn_dropout = 0.5
    hidden_units = [256, 128, 64]

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    # ========================== Create dataset =======================
    feature_columns, train, test = criteo.build_dataset(mode=mode)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    model = Deep_Crossing(feature_columns, hidden_units)
    # model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/deep_crossing_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(
        learning_rate=learning_rate), metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(
            monitor='val_auc', patience=2, restore_best_weights=True)],
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


def input_data(dataset, max_sl):
    user = np.array(dataset[:, 0], dtype='int32')
    item = np.array(dataset[:, 1], dtype='int32')
    hist = dataset[:, 2]
    hist_matrix = keras.preprocessing.sequence.pad_sequences(
        hist, maxlen=max_sl, padding='post')

    sl = np.array(dataset[:, 3], dtype='int32')
    y = np.array(dataset[:, 4], dtype='float32')

    return user, item, hist_matrix, sl, y


def train_din(mode=1):
    hidden_unit = 64
    batch_size = 32
    learning_rate = 1
    epochs = 50

    with open(electronics.pkl_dataset, 'rb') as f:
        train_set = np.array(pickle.load(f))
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, max_sl = pickle.load(f)
    train_user, train_item, train_hist, train_sl, train_y = input_data(
        train_set, max_sl)
    # Tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_grads=False,
        write_images=True,
        embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None, update_freq=500
    )
    # model checkpoint
    check_path = 'save/din_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    checkpoint = keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
                                                 verbose=1, period=1)

    model = DIN(user_count, item_count, cate_count, cate_list, hidden_unit)
    model.summary()
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, decay=0.1)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=optimizer, metrics=[keras.metrics.AUC()])
    model.fit(
        [train_user, train_item, train_hist, train_sl],
        train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        #callbacks=[tensorboard, checkpoint]
    )
