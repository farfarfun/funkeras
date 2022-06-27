from tensorflow import keras
from tensorflow.keras import layers

from notekeras.backend import backend as K
from notekeras.layers import SelfSum, MaskFlatten

Layer = layers.Layer
Dense = layers.Dense


class WideDeepComponent(Layer):
    def __init__(self,
                 name,
                 deep_shape=None,
                 deep_activation='relu'
                 ):
        super(WideDeepComponent, self).__init__(name=name)

        if deep_shape is None:
            deep_shape = [64, 64, 32]

        self.deep_shape = deep_shape
        self.deep_activation = deep_activation

        self.attention_layer2 = None
        self.feed_forward_layer2 = None

        self.supports_masking = True

        self.deep_dense = []
        self._build()

    def _build(self):
        for shape in self.deep_shape:
            dense = Dense(shape, activation=self.deep_activation)
            self.deep_dense.append(dense)
        self.liner = Dense(1, activation='sigmoid')

    def __call__(self, inputs, **kwargs):
        input_deep, input_wide = inputs
        deep = input_deep

        for dense in self.deep_dense:
            deep = dense(deep)
        concat = K.concatenate(deep, input_wide)
        out = self.liner(concat)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'deep_shape': self.deep_shape,
            'deep_activation': self.deep_activation,
        }
        return config


class DeepFM(Layer):
    """
    主要参考 https://blog.csdn.net/songbinxu/article/details/80151814
    """

    def __init__(self,
                 name='DeepFM',
                 deep_shape=None,
                 deep_activation='relu'
                 ):
        super(DeepFM, self).__init__(name=name)

        if deep_shape is None:
            deep_shape = [128, 64, 32, 1]

        self.deep_shape = deep_shape
        self.deep_activation = deep_activation

        self.attention_layer2 = None
        self.feed_forward_layer2 = None

        self.supports_masking = True

        self.deep_dense = []
        self._build()

    def _build(self):

        for shape in self.deep_shape:
            dense = keras.layers.Dense(shape, activation=self.deep_activation,
                                       name='{}-deep-dense{}'.format(self.name, len(self.deep_dense)))
            self.deep_dense.append(dense)

        self.liner = keras.layers.Dense(1, activation='sigmoid')

        self.self_sum1 = SelfSum(axis=1, name='{}-SelfSum1'.format(self.name))
        self.self_sum2 = SelfSum(axis=1, name='{}-SelfSum2'.format(self.name))
        self.self_sum3 = SelfSum(axis=1, name='{}-SelfSum3'.format(self.name))

        self.multiple1 = keras.layers.Multiply(name='{}-multiple1'.format(self.name))
        self.multiple2 = keras.layers.Multiply(name='{}-multiple2'.format(self.name))
        self.multiple3 = keras.layers.Multiply(name='{}-multiple3'.format(self.name))

        self.subtract = keras.layers.Subtract(name='{}-subtract'.format(self.name))

    def __call__(self, inputs, **kwargs):
        emb, y_first_order = inputs

        '''compute'''
        summed_features_emb = self.self_sum1(emb)  # None * K
        summed_features_emb_square = self.multiple1([summed_features_emb, summed_features_emb])  # None * K

        squared_features_emb = self.multiple2([emb, emb])  # None * 9 * K
        squared_sum_features_emb = self.self_sum2(squared_features_emb)  # Non * K

        sub = self.subtract([summed_features_emb_square, squared_sum_features_emb])  # None * K
        sub = keras.layers.Lambda(lambda x: x * 0.5)(sub)  # None * K

        # sub = FM(8, use_weight=False, use_bias=False)(emb)

        # y_second_order = self.self_sum3(sub)  # None * 1
        y_second_order = sub

        '''deep parts'''
        y_deep = MaskFlatten()(emb)  # None*(6*K)

        for dense in self.deep_dense:
            y_deep = keras.layers.Dropout(0.5)(dense(y_deep))

        '''deepFM'''
        y = keras.layers.Concatenate(axis=1)([y_first_order, y_second_order, y_deep])
        y = keras.layers.Dense(1, activation='sigmoid')(y)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'deep_shape': self.deep_shape,
            'deep_activation': self.deep_activation,
        }
        return config
