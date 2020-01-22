import numpy as np
import tensorflow as tf

from notekeras.backend import backend as K
from notekeras.backend import keras, layers, Dense

Layer = keras.layers.Layer

__all__ = ['MaskFlatten',
           'SelfMean', 'SelfSum',
           'FactorizationMachine', 'FM', 'CrossLayer']


class SelfMean(Layer):
    """
     其功能是对2d或3d的tensor，指定一个axis进行求均值。例如[100,5,6]的矩阵，指定axis=1求均值，会变成[100,6]大小的矩阵。
    """

    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(SelfMean, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        if mask is not None:
            if K.ndim(x) != K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0, 2, 1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
        else:
            return K.mean(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i != self.axis:
                output_shape.append(input_shape[i])
        return tuple(output_shape)


class SelfSum(Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(SelfSum, self).__init__(**kwargs)

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
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


class MaskFlatten(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskFlatten, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return mask
        return K.batch_flatten(mask)

    def call(self, inputs, mask=None):
        return K.batch_flatten(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], np.prod(input_shape[1:])


class FactorizationMachine(Layer):
    def __init__(self,
                 output_dim=30,
                 name='FM',
                 activation="relu",
                 use_weight=True,
                 use_bias=True,
                 **kwargs):
        super(FactorizationMachine, self).__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.activate = activation
        self.user_weight = use_weight
        self.use_bias = use_bias
        self.weight = self.bias = self.kernel = None
        self.activate_layer = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        if self.user_weight:
            self.weight = self.add_weight(name='weight',
                                          shape=(input_shape[1], self.output_dim),
                                          initializer='glorot_uniform',
                                          trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.output_dim,),
                                        initializer='zeros',
                                        trainable=True)

        self.activate_layer = keras.activations.get(self.activate)
        super(FactorizationMachine, self).build(input_shape)

    def call(self, inputs, **kwargs):
        xv_a = K.square(K.dot(inputs, self.kernel))
        xv_b = K.dot(K.square(inputs), K.square(self.kernel))

        p = 0.5 * K.sum(xv_a - xv_b, 1)

        xv = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)

        res = xv
        if self.user_weight:
            res = res + K.dot(inputs, self.weight)
        if self.use_bias:
            res = res + self.bias

        output = K.reshape(res, (-1, self.output_dim))

        if self.activate_layer is not None:
            output = self.activate_layer(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim


FM = FactorizationMachine


class CrossLayer(Layer):
    """
    https://keras.io/layers/writing-your-own-keras-layers/
    """

    def __init__(self,
                 num_layer,
                 name='Cross',
                 **kwargs):
        super(CrossLayer, self).__init__(name=name, **kwargs)
        self.num_layer = num_layer
        self.input_dim = self.W = self.bias = None

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.W = []
        self.bias = []
        for i in range(self.num_layer):
            self.W.append(self.add_weight(shape=[1, self.input_dim],
                                          initializer='glorot_uniform',
                                          name='w_' + str(i),
                                          trainable=True))
            self.bias.append(self.add_weight(shape=[1, self.input_dim],
                                             initializer='zeros', name='b_' + str(i),
                                             trainable=True))
        super(CrossLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = K.reshape(inputs, (-1, 1, self.input_dim))

        cross = inputs
        for i in range(self.num_layer):
            cross = layers.Lambda(lambda x: layers.Add()(
                [K.sum(self.W[i] * K.batch_dot(K.reshape(x, (-1, self.input_dim, 1)), inputs), 1, keepdims=True),
                 self.bias[i], inputs]))(cross)

        cross = K.reshape(cross, (-1, self.input_dim))
        return cross

    def compute_output_shape(self, input_shape):
        return input_shape


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
                                           name='{}-encode-{}'.format(self.name, index),
                                           activation=self.hidden_activation
                                           ))
            self.decode_layer.insert(0, Dense(layer_num,
                                              name='{}-decode-{}'.format(self.name, index_max - index),
                                              activation=self.hidden_activation
                                              ))
        self.encode_layer.append(Dense(self.hidden_layer[-1],
                                       name='{}-encode-{}'.format(self.name, index_max),
                                       activation=self.hidden_activation
                                       ))
        self.decode_layer.append(Dense(input_shape[-1],
                                       name='{}-decode-{}'.format(self.name, index_max),
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
