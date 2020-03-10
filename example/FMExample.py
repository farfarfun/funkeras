import numpy as np
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model

from notekeras.layer.core import FM


def tes_model(x_train, x_test, y_train, y_test, train=False):
    inp = Input(shape=[6])
    x = Dense(50, activation='relu')(inp)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=x)

    if train:
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=2,
                  validation_data=(x_test, y_test))

    print(model.summary())
    return model


def fm_model(x_train, x_test, y_train, y_test, train=False):
    inp = Input(shape=[6])
    x = FM(50, activation='relu')(inp)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=x)

    if train:
        model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=2,
                  validation_data=(x_test, y_test))

    print(model.summary())
    return model


# X = np.linspace(-2, 6, 200)
# np.random.shuffle(X)
# Y = 0.5 * X + 2 + 0.15 * np.random.randn(200, )
#
# # plot data
# plt.scatter(X, Y)
# plt.show()
def get_data(size=100000):
    x = np.random.rand(size, 6)
    x[:, 4] = x[:, 1] * x[:, 3]
    x[:, 5] = x[:, 1] * x[:, 2]
    w = np.random.rand(1, 6)
    y = np.matmul(x, np.transpose(w))

    d = int(size * 0.8)
    print((size, d))
    x_train, y_train = x[:d], y[:d]
    x_test, y_test = x[d:], y[d:]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = get_data()

tes_model(x_train, x_test, y_train, y_test, train=True)
fm_model(x_train, x_test, y_train, y_test, train=True)
