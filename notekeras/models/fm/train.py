import datetime
import pickle

from notedata.dataset.datas import CriteoData, ElectronicsData
from tensorflow.keras import (callbacks, losses, metrics, optimizers,
                              preprocessing)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from notekeras.models.fm.model import *

from .model import *

criteo = CriteoData()
electronics = ElectronicsData()


def download(mode=1):
    criteo.download(mode=mode)
    electronics.download()


def train_all(mode=1, name='FM'):
    learning_rate = 0.001
    batch_size = 4096
    epochs = 10

    feature_columns, train, test = criteo.build_dataset(mode=mode)

    # ============================Build Model==========================
    model = None
    if name == 'FM':
        model = FM(feature_columns=feature_columns, k=10)
    elif name == 'AFM':
        mode = 'avg'  # 'max', 'avg'
        model = AFM(feature_columns, mode)
    elif name == 'FFM':
        model = FFM(feature_columns=feature_columns, k=10)
    elif name == "NFM":
        model = NFM(feature_columns, hidden_units=[
                    256, 128, 64], dnn_dropout=0.5)
    elif name == "DeepFM":
        model = DeepFM(feature_columns, k=10, hidden_units=[
                       256, 128, 64], dnn_dropout=0.5)
    elif name == "xDeepFM":
        model = xDeepFM(feature_columns, hidden_units=[
                        256, 128, 64], cin_size=[128, 128])
    elif name == "MF":
        model = MF(feature_columns, implicit=False, use_bias=True)
    elif name == "WideDeep":
        model = WideDeep(feature_columns, hidden_units=[
                         256, 128, 64], dnn_dropout=0.5)
    elif name == "DCN":
        model = DCN(feature_columns, [256, 128, 64], dnn_dropout=0.5)
    elif name == "PNN":
        model = PNN(feature_columns, hidden_units=[
                    256, 128, 64], dnn_dropout=0.5)
    elif name == "Deep_Crossing":
        model = Deep_Crossing(feature_columns, hidden_units=[256, 128, 64])

    if model is not None:
        model.compile(loss=binary_crossentropy, optimizer=Adam(
            learning_rate=learning_rate), metrics=[AUC()])

    # model.summary()
    # ==============================Fit==============================

    if model is not None:
        plot_model(model, to_file=criteo.path_root +
                   '/'+name+'.png', show_shapes=True)
        plot_model(model, to_file=criteo.path_root +
                   '/'+name+'-extend.png',
                   show_shapes=True, expand_nested=True)

        check_path = criteo.path_root+'/save/{name}_weights.epoch_{epoch}.val_loss.ckpt'.format(
            epoch=epochs, name=name)
        checkpoint = callbacks.ModelCheckpoint(
            check_path, save_weights_only=True, verbose=1, period=5)
        early = EarlyStopping(monitor='val_loss',
                              patience=2, restore_best_weights=True)
        train_x, train_y = train
        test_x, test_y = test
        model.fit(train_x, train_y, epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.1,
                  callbacks=[
                      # checkpoint,
                      early],
                  )

        print('test AUC: %f' % model.evaluate(test_x, test_y)[1])


def input_data(dataset, max_sl):
    user = np.array(dataset[:, 0], dtype='int32')
    item = np.array(dataset[:, 1], dtype='int32')
    hist = dataset[:, 2]
    hist_matrix = preprocessing.sequence.pad_sequences(
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
    test_user, test_item, test_hist, test_sl, test_y = input_data(
        test_set, max_sl)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/' + current_time
    tensorboard = callbacks.TensorBoard(log_dir=log_dir,
                                        histogram_freq=1,
                                        write_graph=True,
                                        write_grads=False,
                                        write_images=True,
                                        embeddings_freq=0, embeddings_layer_names=None,
                                        embeddings_metadata=None, embeddings_data=None, update_freq=500
                                        )
    # model checkpoint
    check_path = 'save/din_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    checkpoint = callbacks.ModelCheckpoint(
        check_path, save_weights_only=True, verbose=1, period=1)

    model = DIN(user_count, item_count, cate_count, cate_list, hidden_unit)
    model.summary()
    optimizer = optimizers.SGD(learning_rate=learning_rate, decay=0.1)
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizer, metrics=[metrics.AUC()])
    model.fit([train_user, train_item, train_hist, train_sl],
              train_y,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.1,
              # callbacks=[tensorboard, checkpoint]
              )
    print('test AUC: %f' % model.evaluate(
        [test_user, test_item, test_hist, test_sl], test_y))
