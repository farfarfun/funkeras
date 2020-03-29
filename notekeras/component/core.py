from tensorflow import keras

from notekeras.backend import keras
from notekeras.layer.attention import MultiHeadAttention
from notekeras.layer.feed_forward import FeedForward
from notekeras.layer.normalize import LayerNormalization

Layer = keras.layers.Layer
Model = keras.models.Model


class Component(keras.models.Model):
    def __init__(self,
                 as_model=False,
                 inputs=None,
                 as_layer=False,
                 layer_depth=1,
                 *args, **kwargs):
        """
        :param layer_depth:
        :param as_model:组件是否当做一个模型返回，当模型的时候，输入inputs必须非空
        :param inputs:模型输入，当as_model为True时传入
        :param args:
        :param kwargs:
        """
        super(Component, self).__init__(*args, **kwargs)
        self.as_model = as_model

        self.layer_depth = layer_depth
        self.input_layer = inputs

        if self.as_model:
            if inputs is None:
                raise BaseException("when as_model is True, inputs must not be None")
            self.outputs = self._build(inputs)
            self.output_layer = self.outputs
            super(Component, self).__init__(inputs=inputs, outputs=self.outputs, *args, **kwargs)

    def _build(self, inputs):

        return inputs

    def __call__(self, inputs, **kwargs):
        if self.as_model:
            return super(Component, self).__call__(inputs, **kwargs)

        if self.layer_depth <= 0:
            return super(Component, self).__call__(inputs=inputs, **kwargs)
        else:
            return self.call(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        return self._build(inputs)


class WrapCodeComponent(Component):
    def __init__(self,
                 name,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 history_only=False,
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',
                 is_list=False,
                 use_attention=True,
                 *args,
                 **kwargs
                 ):
        # super(WrapCodeComponent, self).__init__(name=name)
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.attention_activation = attention_activation
        self.history_only = history_only
        self.feed_forward_activation = feed_forward_activation
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation
        self.is_list = is_list
        self.use_attention = use_attention

        self.dropout_layer1 = self.dropout_layer2 = self.adapter = None
        self.add_layer = self.adapter = self.normal_layer = None

        self.supports_masking = True
        self.attention_layer = None

        super(WrapCodeComponent, self).__init__(name=name, *args, **kwargs)

    def _build(self, inputs):
        if self.use_attention:
            self.attention_layer = MultiHeadAttention(name='%s_Attention' % self.name,
                                                      head_num=self.head_num,
                                                      activation=self.attention_activation,
                                                      history_only=self.history_only,
                                                      trainable=self.trainable,
                                                      )
        else:
            self.attention_layer = FeedForward(name='%s_FeedForward' % self.name,
                                               units=self.hidden_dim,
                                               activation=self.feed_forward_activation,
                                               trainable=self.trainable,
                                               )

        if self.dropout_rate > 0.0:
            self.dropout_layer1 = keras.layers.Dropout(rate=self.dropout_rate, name='%s_Dropout' % self.name, )

        if self.use_adapter:
            self.adapter = FeedForward(units=self.adapter_units,
                                       activation=self.adapter_activation,
                                       kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
                                       name='%s_Adapter' % self.name,
                                       )
            self.dropout_layer2 = keras.layers.Add(name='%s_Adapter_Add' % self.name)
        #
        self.add_layer = keras.layers.Add(name='%s_Add' % self.name)

        # 正则化
        self.normal_layer = LayerNormalization(trainable=self.trainable, name='%s_Norm' % self.name, )

        build_output = self.attention_layer(inputs)

        if self.dropout_rate > 0.0:
            dropout_layer = self.dropout_layer1(build_output)
        else:
            dropout_layer = build_output

        if isinstance(inputs, list):
            inputs = inputs[0]
        if self.use_adapter:
            adapter = self.adapter(dropout_layer)
            dropout_layer = self.dropout_layer2([dropout_layer, adapter])
        #
        add_layer = self.add_layer([inputs, dropout_layer])

        # 正则化
        normal_layer = self.normal_layer(add_layer)

        return normal_layer

    def compute_output_shape(self, input_shape):
        return input_shape
