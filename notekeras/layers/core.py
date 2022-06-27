import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, layers
from tensorflow.keras.layers import Dense, Dropout, Layer
from tensorflow.keras.regularizers import l2


class SelfMean(Layer):
    """
    其功能是对2d或3d的tensor，指定一个axis进行求均值。例如[100,5,6]的矩阵，指定axis=1求均值，会变成[100,6]大小的矩阵。
    """

    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(SelfMean, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None):
        if mask is not None:
            if K.ndim(x) != K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0, 2, 1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)

        return K.mean(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i != self.axis:
                output_shape.append(input_shape[i])
        return tuple(output_shape)

    def get_config(self):
        config = {"axis": self.axis}
        config.update(super(SelfMean, self).get_config())
        return config


class SelfSum(Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(SelfSum, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, x, mask=None, *args, **kwargs):
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            if K.ndim(x) != K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0, 2, 1])
            x = x * mask
            if K.ndim(x) == 2:
                x = K.expand_dims(x)
            return K.sum(x, axis=self.axis)
        else:
            if K.ndim(x) == 2:
                x = K.expand_dims(x)
            return K.sum(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i != self.axis:
                output_shape.append(input_shape[i])
        if len(output_shape) == 1:
            output_shape.append(1)
        return tuple(output_shape)

    def get_config(self):
        config = {"axis": self.axis}
        config.update(super(SelfSum, self).get_config())
        return config


class MaskFlatten(Layer):
    def __init__(self, *args, **kwargs):
        self.supports_masking = True
        super(MaskFlatten, self).__init__(*args, **kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask
        return K.batch_flatten(mask)

    def call(self, inputs, mask=None):
        return K.batch_flatten(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], np.prod(input_shape[1:])


class AutoEncoder(Layer):
    """
    https://keras.io/layers/writing-your-own-keras-layers/
    """

    def __init__(self,
                 hidden_layer=None,
                 name='Cross',
                 is_layer=False,
                 hidden_activation='relu',
                 output_activation='tanh',
                 **kwargs):
        super(AutoEncoder, self).__init__(name=name, **kwargs)
        if hidden_layer is None:
            hidden_layer = [128, 64, 32]

        self.is_layer = is_layer
        self.hidden_layer = hidden_layer
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.encode_layer = self.decode_layer = None

    def build(self, input_shape):

        self.encode_layer = []
        self.decode_layer = []

        index = 0
        index_max = len(self.hidden_layer)
        for layer_num in self.hidden_layer[:-1]:
            index += 1
            self.encode_layer.append(Dense(layer_num,
                                           name='{}-encode-{}'.format(
                                               self.name, index),
                                           activation=self.hidden_activation
                                           ))
            self.decode_layer.insert(0, Dense(layer_num,
                                              name='{}-decode-{}'.format(
                                                  self.name, index_max - index),
                                              activation=self.hidden_activation
                                              ))
        self.encode_layer.append(Dense(self.hidden_layer[-1],
                                       name='{}-encode-{}'.format(
                                           self.name, index_max),
                                       activation=self.hidden_activation
                                       ))
        self.decode_layer.append(Dense(input_shape[-1],
                                       name='{}-decode-{}'.format(
                                           self.name, index_max),
                                       activation=self.output_activation
                                       ))

        super(AutoEncoder, self).build(input_shape)

    def call(self, inputs, **kwargs):
        layer = inputs
        for encoder in self.encode_layer:
            layer = encoder(layer)

        for decoder in self.decode_layer:
            layer = decoder(layer)

        return layer

    def __call__(self, inputs, **kwargs):
        if self.is_layer:
            return super(AutoEncoder, self).__call__(inputs, **kwargs)
        else:
            self.build(inputs.shape)
            return self.call(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class CrossLayer(Layer):
    def __init__(self, layer_num=5, reg_weight=1e-4, reg_bias=1e-4, *args, **kwargs):
        super(CrossLayer, self).__init__(*args, **kwargs)
        self.layer_num = layer_num
        self.reg_weight = reg_weight
        self.reg_bias = reg_bias
        self.kernels = self.biases = None

    def build(self, input_shape):
        dim = int(input_shape[1])
        self.kernels = [self.add_weight(name='weight_' + str(i),
                                        shape=(dim, 1),
                                        initializer='random_uniform',
                                        regularizer=l2(self.reg_weight),
                                        trainable=True, ) for i in range(self.layer_num)]

        self.biases = [self.add_weight(name='bias_' + str(i),
                                       shape=(dim, 1),
                                       initializer='random_uniform',
                                       regularizer=l2(self.reg_bias),
                                       trainable=True, ) for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        # (None, dim, 1)
        x_0 = tf.expand_dims(inputs, axis=2)
        # (None, dim, 1)
        x_l = x_0
        for i in range(self.layer_num):
            # (None, dim, dim)
            x_l1 = tf.tensordot(x_l, self.kernels[i], axes=[1, 0])
            # (None, dim, 1)
            x_l = tf.matmul(x_0, x_l1) + self.biases[i] + x_l

        # (None, dim)
        x_l = tf.squeeze(x_l, axis=2)
        return x_l


class Linear(Dense):
    """
    Linear Part
    """

    def __init__(self, *args, **kwargs):
        super(Linear, self).__init__(1, activation=None, *args, **kwargs)


class DNN(Layer):
    """
        Deep Neural Network
        """

    def __init__(self,
                 hidden_units,
                 activation='relu',
                 dropout=0.,
                 name='DNN',
                 *args,
                 **kwargs):
        """
        :param hidden_units: A list. Neural network hidden units.
        :param activation: A string. Activation function of dnn.
        :param dropout: A scalar. Dropout number.
        """
        super(DNN, self).__init__(*args, **kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = dropout
        self.dnn_network = None
        self.args = args
        self.kwargs = kwargs

    def build(self, input_shape):
        self.dnn_network = [Dense(units=unit, activation=self.activation, *self.args, **self.kwargs)
                            for unit in self.hidden_units]
        self.dropout = Dropout(self.dropout)
        super(DNN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x
