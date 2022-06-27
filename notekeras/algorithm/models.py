from notekeras.layers.attention import MultiHeadAttention
from notekeras.layers.embedding import (EmbeddingRet, EmbeddingSim,
                                        TrigPosEmbedding)
from notekeras.layers.feed_forward import FeedForward
from notekeras.layers.normalize import LayerNormalization
from tensorflow import keras

__all__ = []


# VGG-16, ZF-Net,Alex-Net,LeNet,Google-Net,ResNet, DenseNet-50

def get_custom_objects():
    return {
        'LayerNormalization': LayerNormalization,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForward': FeedForward,
        'TrigPosEmbedding': TrigPosEmbedding,
        'EmbeddingRet': EmbeddingRet,
        'EmbeddingSim': EmbeddingSim,
    }


class WrapLayer0(keras.models.Model):
    def __init__(self,
                 name,
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',
                 is_list=False,
                 ):
        super(WrapLayer0, self).__init__(name=name)
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation

        self.dropout_layer1 = self.dropout_layer2 = self.adapter = None
        self.add_layer = self.adapter = self.normal_layer = None
        self.is_list = is_list

        self.supports_masking = True
        self._build()

    def _build(self):
        if self.dropout_rate > 0.0:
            self.dropout_layer1 = keras.layers.Dropout(
                rate=self.dropout_rate, name='%s-Dropout' % self.name, )
            self.layers.append(self.dropout_layer1)

        if self.use_adapter:
            self.adapter = FeedForward(units=self.adapter_units,
                                       activation=self.adapter_activation,
                                       kernel_initializer=keras.initializers.TruncatedNormal(
                                           mean=0.0, stddev=0.001),
                                       name='%s-Adapter' % self.name,
                                       )
            self.layers.append(self.adapter)
            self.dropout_layer2 = keras.layers.Add(
                name='%s-Adapter-Add' % self.name)
            self.layers.append(self.dropout_layer2)
        #
        self.add_layer = keras.layers.Add(name='%s-Add' % self.name)
        self.layers.append(self.add_layer)

        # 正则化
        self.normal_layer = LayerNormalization(
            trainable=self.trainable, name='%s-Norm' % self.name, )
        self.layers.append(self.normal_layer)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        input_layer, build_output = inputs

        if self.dropout_rate > 0.0:
            dropout_layer = self.dropout_layer1(build_output)
        else:
            dropout_layer = build_output

        if isinstance(input_layer, list):
            input_layer = input_layer[0]
        if self.use_adapter:
            adapter = self.adapter(dropout_layer)
            dropout_layer = self.dropout_layer2([dropout_layer, adapter])
        #
        add_layer = self.add_layer([input_layer, dropout_layer])

        # 正则化
        normal_layer = self.normal_layer(add_layer)
        return normal_layer

    def get_config(self):

        config = {
            'layers': [],
            'dropout_rate': self.dropout_rate,
            'trainable': self.trainable,
            'use_adapter': self.use_adapter,

            'adapter_units': self.adapter_units,
        }
        for layer in self.layers:
            config['layers'].append({
                'class_name': layer.__class__.__name__,
                'config': layer.get_config,
            })

        return config
