from notekeras.backend import keras
from notekeras.layer.attention import MultiHeadAttention
from notekeras.layer.feed_forward import FeedForward
from notekeras.layer.normalize import LayerNormalization

Layer = keras.layers.Layer
Model = keras.models.Model

__all__ = ['get_custom_objects',
           'WrapCodeLayer', 'EncoderLayer', 'DecoderLayer',
           'WrapCodeModel', 'EncoderModel', 'DecoderModel',
           'EncoderList', 'DecoderList']


def get_custom_objects():
    return {
        'WrapCodeLayer': WrapCodeLayer,
        'EncoderLayer': EncoderLayer,
        'DecoderLayer': DecoderLayer,
        'EncoderComponent': EncoderList,
        'DecoderComponent': DecoderList
    }


class WrapCodeLayer(Layer):
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
                 as_layer=False):
        super(WrapCodeLayer, self).__init__(name=name)
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
        self.as_layer = as_layer

        self.dropout_layer1 = self.dropout_layer2 = self.adapter = None
        self.add_layer = self.adapter = self.normal_layer = None

        self.supports_masking = True
        self.attention_layer = None

        self._network_nodes = []
        self._build()

        super(WrapCodeLayer, self).__init__(name=name)

    def _build(self):
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

    def __call__(self, inputs, **kwargs):
        if self.as_layer:
            return super(WrapCodeLayer, self).__call__(inputs=inputs, **kwargs)
        else:
            return self.call(inputs, **kwargs)

    def call(self, input_layer, **kwargs):
        build_output = self.attention_layer(input_layer)

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

    def compute_output_shape(self, input_shape):
        return input_shape


class EncoderLayer(Layer):
    def __init__(self,
                 name,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',
                 as_layer=True
                 ):
        super(EncoderLayer, self).__init__(name=name)

        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation
        self.as_layer = as_layer
        self.attention_layer2 = None
        self.feed_forward_layer2 = None

        self.supports_masking = True
        self._build()

    def _build(self):
        self.attention_layer2 = WrapCodeModel(name='%s_MultiHeadSelfAttention' % self.name,
                                              head_num=self.head_num,
                                              hidden_dim=self.hidden_dim,
                                              attention_activation=self.attention_activation,
                                              feed_forward_activation=self.feed_forward_activation,
                                              history_only=False,
                                              dropout_rate=self.dropout_rate,
                                              trainable=self.trainable,
                                              use_adapter=self.use_adapter,
                                              adapter_units=self.adapter_units,
                                              adapter_activation=self.adapter_activation,
                                              use_attention=True,
                                              as_layer=self.as_layer
                                              )

        self.feed_forward_layer2 = WrapCodeModel(name='%s_FeedForward' % self.name,
                                                 head_num=self.head_num,
                                                 hidden_dim=self.hidden_dim,
                                                 attention_activation=self.attention_activation,
                                                 feed_forward_activation=self.feed_forward_activation,
                                                 history_only=False,
                                                 dropout_rate=self.dropout_rate,
                                                 trainable=self.trainable,
                                                 use_adapter=self.use_adapter,
                                                 adapter_units=self.adapter_units,
                                                 adapter_activation=self.adapter_activation,
                                                 use_attention=False,
                                                 as_layer=self.as_layer
                                                 )

    def __call__(self, inputs, **kwargs):
        if self.as_layer:
            return super(EncoderLayer, self).__call__(inputs=inputs, **kwargs)
        else:
            return self.call(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        att2 = self.attention_layer2(inputs)
        feed2 = self.feed_forward_layer2(att2)

        return feed2

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'dropout_rate': self.dropout_rate,
            'trainable': self.trainable,
            'use_adapter': self.use_adapter,
            'adapter_units': self.adapter_units,
        }

        return config


class DecoderLayer(Layer):
    def __init__(self,
                 name,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',
                 as_layer=True
                 ):
        super(DecoderLayer, self).__init__(name=name)
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation
        self.as_layer = as_layer

        self.self_attention_layer2 = None
        self.feed_forward_layer2 = None
        self.query_attention_layer2 = None
        self.supports_masking = True
        self._build()

    def _build(self):
        self.self_attention_layer2 = WrapCodeLayer(name='%s_SelfAttention' % self.name,
                                                   head_num=self.head_num,
                                                   hidden_dim=self.hidden_dim,
                                                   attention_activation=self.attention_activation,
                                                   feed_forward_activation=self.feed_forward_activation,
                                                   history_only=True,

                                                   dropout_rate=self.dropout_rate,
                                                   trainable=self.trainable,
                                                   use_adapter=self.use_adapter,
                                                   adapter_units=self.adapter_units,
                                                   adapter_activation=self.adapter_activation,
                                                   use_attention=True,
                                                   as_layer=self.as_layer
                                                   )

        self.query_attention_layer2 = WrapCodeLayer(name='%s_QueryAttention' % self.name,
                                                    head_num=self.head_num,
                                                    hidden_dim=self.hidden_dim,
                                                    attention_activation=self.attention_activation,
                                                    feed_forward_activation=self.feed_forward_activation,
                                                    history_only=False,
                                                    dropout_rate=self.dropout_rate,
                                                    trainable=self.trainable,
                                                    use_adapter=self.use_adapter,
                                                    adapter_units=self.adapter_units,
                                                    adapter_activation=self.adapter_activation,
                                                    is_list=True,
                                                    use_attention=True,
                                                    as_layer=self.as_layer
                                                    )

        self.feed_forward_layer2 = WrapCodeLayer(name='%s_FeedForward' % self.name,
                                                 head_num=self.head_num,
                                                 hidden_dim=self.hidden_dim,
                                                 attention_activation=self.attention_activation,
                                                 feed_forward_activation=self.feed_forward_activation,
                                                 history_only=True,
                                                 dropout_rate=self.dropout_rate,
                                                 trainable=self.trainable,
                                                 use_adapter=self.use_adapter,
                                                 adapter_units=self.adapter_units,
                                                 adapter_activation=self.adapter_activation,
                                                 use_attention=False,
                                                 as_layer=self.as_layer
                                                 )

    def __call__(self, inputs, **kwargs):
        if self.as_layer:
            return super(DecoderLayer, self).__call__(inputs=inputs, **kwargs)
        else:
            return self.call(inputs, **kwargs)

    def call(self, inputs, **kwargs):
        input_layer, encoded_layer = inputs

        self_att2 = self.self_attention_layer2(input_layer)

        query_att2 = self.query_attention_layer2([self_att2, encoded_layer, encoded_layer])

        feed2 = self.feed_forward_layer2(query_att2)

        return feed2

    def get_config(self):
        config = {
            'dropout_rate': self.dropout_rate,
            'trainable': self.trainable,
            'use_adapter': self.use_adapter,
            'adapter_units': self.adapter_units,
        }
        return config


class WrapCodeModel(Model):
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
                 input_shape=None,
                 **kwargs
                 ):
        super(WrapCodeModel, self).__init__(name=name, **kwargs)
        self.input_sh = input_shape
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

        self._network_nodes = []
        self.input_layer = self.output_layer = None

        self._build(input_shape)

        super(WrapCodeModel, self).__init__(inputs=self.input_layer,
                                            outputs=self.output_layer,
                                            name=name,
                                            **kwargs)

    def _build(self, input_shape):
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

        # def call(self, input_layer, **kwargs):
        if self.is_list:
            input_layer = [keras.layers.Input(shape=input_shape, name='{}_Encoder-Input-key'.format(self.name)),
                           keras.layers.Input(shape=input_shape, name='{}_Encoder-Input-query'.format(self.name)),
                           keras.layers.Input(shape=input_shape, name='{}_Encoder-Input-value'.format(self.name))
                           ]
        else:
            input_layer = keras.layers.Input(shape=input_shape, name='{}_Encoder-Input'.format(self.name))
        build_output = self.attention_layer(input_layer)

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

        self.input_layer = input_layer
        self.output_layer = normal_layer
        return normal_layer

    def compute_output_shape(self, input_shape):
        return input_shape


class EncoderModel(Model):
    def __init__(self,
                 head_num,
                 hidden_dim,
                 name='Encode',
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',
                 input_shape=None
                 ):
        super(EncoderModel, self).__init__(name=name)

        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation
        self.attention_layer2 = None
        self.feed_forward_layer2 = None

        self.supports_masking = True

        self.input_layer = self.output_layer = None

        self._build(input_shape)

        super(EncoderModel, self).__init__(inputs=self.input_layer,
                                           outputs=self.output_layer,
                                           name=name,
                                           trainable=trainable)

    def _build(self, input_shape):
        self.attention_layer2 = WrapCodeModel(name='%s_MultiHeadSelfAttention' % self.name,
                                              head_num=self.head_num,
                                              hidden_dim=self.hidden_dim,
                                              attention_activation=self.attention_activation,
                                              feed_forward_activation=self.feed_forward_activation,
                                              history_only=False,
                                              dropout_rate=self.dropout_rate,
                                              trainable=self.trainable,
                                              use_adapter=self.use_adapter,
                                              adapter_units=self.adapter_units,
                                              adapter_activation=self.adapter_activation,
                                              use_attention=True,
                                              input_shape=input_shape
                                              )

        self.feed_forward_layer2 = WrapCodeModel(name='%s_FeedForward' % self.name,
                                                 head_num=self.head_num,
                                                 hidden_dim=self.hidden_dim,
                                                 attention_activation=self.attention_activation,
                                                 feed_forward_activation=self.feed_forward_activation,
                                                 history_only=False,
                                                 dropout_rate=self.dropout_rate,
                                                 trainable=self.trainable,
                                                 use_adapter=self.use_adapter,
                                                 adapter_units=self.adapter_units,
                                                 adapter_activation=self.adapter_activation,
                                                 use_attention=False,
                                                 input_shape=input_shape
                                                 )

        input_layer = keras.layers.Input(shape=input_shape, name='{}-Encoder-Input'.format(self.name))
        att2 = self.attention_layer2(input_layer)
        feed2 = self.feed_forward_layer2(att2)

        self.input_layer = input_layer
        self.output_layer = feed2
        return feed2

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'dropout_rate': self.dropout_rate,
            'trainable': self.trainable,
            'use_adapter': self.use_adapter,
            'adapter_units': self.adapter_units,
        }

        return config


class DecoderModel(Model):
    def __init__(self,
                 head_num,
                 hidden_dim,
                 name='Decode',
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',
                 input_shape=None,
                 as_layer=True,
                 **kwargs
                 ):
        super(DecoderModel, self).__init__(name=name)
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation
        self.as_layer = as_layer
        self.self_attention_layer = None
        self.feed_forward_layer = None
        self.query_attention_layer = None
        self.supports_masking = True

        self.input_layer = self.output_layer = None

        self._build(input_shape)

        super(DecoderModel, self).__init__(inputs=self.input_layer,
                                           outputs=self.output_layer,
                                           name=name,
                                           **kwargs)

    def _build(self, input_shape):
        self.self_attention_layer = WrapCodeModel(name='%s_SelfAttention' % self.name,
                                                  head_num=self.head_num,
                                                  hidden_dim=self.hidden_dim,
                                                  attention_activation=self.attention_activation,
                                                  feed_forward_activation=self.feed_forward_activation,
                                                  history_only=True,
                                                  dropout_rate=self.dropout_rate,
                                                  trainable=self.trainable,
                                                  use_adapter=self.use_adapter,
                                                  adapter_units=self.adapter_units,
                                                  adapter_activation=self.adapter_activation,
                                                  use_attention=True,
                                                  input_shape=input_shape)

        self.query_attention_layer = WrapCodeLayer(name='%s_QueryAttention' % self.name,
                                                   head_num=self.head_num,
                                                   hidden_dim=self.hidden_dim,
                                                   attention_activation=self.attention_activation,
                                                   feed_forward_activation=self.feed_forward_activation,
                                                   history_only=False,
                                                   dropout_rate=self.dropout_rate,
                                                   trainable=self.trainable,
                                                   use_adapter=self.use_adapter,
                                                   adapter_units=self.adapter_units,
                                                   adapter_activation=self.adapter_activation,
                                                   is_list=True,
                                                   use_attention=True,
                                                   as_layer=self.as_layer,
                                                   )

        self.feed_forward_layer = WrapCodeModel(name='%s_FeedForward' % self.name,
                                                head_num=self.head_num,
                                                hidden_dim=self.hidden_dim,
                                                attention_activation=self.attention_activation,
                                                feed_forward_activation=self.feed_forward_activation,
                                                history_only=True,
                                                dropout_rate=self.dropout_rate,
                                                trainable=self.trainable,
                                                use_adapter=self.use_adapter,
                                                adapter_units=self.adapter_units,
                                                adapter_activation=self.adapter_activation,
                                                use_attention=False,
                                                input_shape=input_shape
                                                )

        input_layer = keras.layers.Input(shape=input_shape, name='{}_Decoder_Input'.format(self.name))
        encoded_layer = keras.layers.Input(shape=input_shape, name='{}_Decoder_Input_Encode'.format(self.name))

        self_att2 = self.self_attention_layer(input_layer)
        query_att2 = self.query_attention_layer([self_att2, encoded_layer, encoded_layer])
        feed2 = self.feed_forward_layer(query_att2)

        self.input_layer = [input_layer, encoded_layer]
        self.output_layer = feed2
        return feed2

    def get_config(self):
        config = {
            'dropout_rate': self.dropout_rate,
            'trainable': self.trainable,
            'use_adapter': self.use_adapter,
            'adapter_units': self.adapter_units,
        }
        return config


class EncoderList(Layer):
    def __init__(self,
                 encoder_num,
                 name='Encoder',
                 **kwargs):
        super(EncoderList, self).__init__(name=name)
        self.encoder_num = encoder_num

        self.layers = []

        self._build(**kwargs)

    def _build(self, **kwargs):
        for i in range(self.encoder_num):
            self.layers.append(EncoderModel(name='{}_{}'.format(self.name, i + 1), **kwargs))
        pass

    def __call__(self, inputs, **kwargs):
        input_layer = inputs
        last_layer = input_layer
        for layer in self.layers:
            last_layer = layer(last_layer)
        return last_layer

    def compute_output_shape(self, input_shape):
        return input_shape


class DecoderList(Layer):
    def __init__(self,
                 decoder_num,
                 name='Decoder',
                 **kwargs):
        super(DecoderList, self).__init__(name=name)
        self.decoder_num = decoder_num
        self.layers = []
        self._build(**kwargs)

    def _build(self, **kwargs):
        for i in range(self.decoder_num):
            self.layers.append(DecoderModel(name='{}_{}'.format(self.name, i + 1), **kwargs))

    def __call__(self, inputs, **kwargs):
        input_layer, encoded_layer = inputs
        last_layer = input_layer
        for layer in self.layers:
            last_layer = layer([last_layer, encoded_layer])
        return last_layer

    def compute_output_shape(self, input_shape):
        return input_shape
