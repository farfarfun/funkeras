from notekeras.backend import keras
from notekeras.component import Component
from notekeras.layers.attention import MultiHeadAttention
from notekeras.layers.feed_forward import FeedForward
from notekeras.layers.normalize import LayerNormalization

__all__ = ['get_custom_objects',
           'WrapCodeComponent', 'EncoderComponent', 'DecoderComponent', 'EncoderList', 'DecoderList']


def get_custom_objects():
    return {
        'WrapCodeLayer': WrapCodeComponent,
        'EncoderLayer': EncoderComponent,
        'DecoderLayer': DecoderComponent,
        'EncoderComponent': EncoderList,
        'DecoderComponent': DecoderList
    }


class WrapCodeComponent(Component):
    def __init__(self,
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
        super(WrapCodeComponent, self).__init__()
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

        kwargs['trainable'] = self.trainable

        super(WrapCodeComponent, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        if self.use_attention:
            self.attention_layer = MultiHeadAttention(head_num=self.head_num,
                                                      activation=self.attention_activation,
                                                      history_only=self.history_only,
                                                      trainable=self.trainable,
                                                      )
        else:
            self.attention_layer = FeedForward(units=self.hidden_dim,
                                               activation=self.feed_forward_activation,
                                               trainable=self.trainable,
                                               )

        if self.dropout_rate > 0.0:
            self.dropout_layer1 = keras.layers.Dropout(rate=self.dropout_rate,)

        if self.use_adapter:
            self.adapter = FeedForward(units=self.adapter_units,
                                       activation=self.adapter_activation,
                                       kernel_initializer=keras.initializers.TruncatedNormal(
                                           mean=0.0, stddev=0.001),
                                       )
            self.dropout_layer2 = keras.layers.Add()

        self.add_layer = keras.layers.Add()

        # 正则化
        self.normal_layer = LayerNormalization(trainable=self.trainable,)

    def call(self, inputs, training=None, mask=None):
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


class EncoderComponent(Component):
    def __init__(self,
                 head_num,
                 hidden_dim,
                 name='encode',
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',
                 input_shape=None,
                 *args, **kwargs):
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation

        self.supports_masking = True

        self.attention_layer2 = self.feed_forward_layer2 = None

        super(EncoderComponent, self).__init__(
            name=name, trainable=trainable, *args, **kwargs)

    def build(self, input_shape):
        self.attention_layer2 = WrapCodeComponent(name="attention_layer",
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
                                                  layer_depth=self.layer_depth - 1,
                                                  )
        self.attention_layer2.run_eagerly = False

        self.feed_forward_layer2 = WrapCodeComponent(name="output", head_num=self.head_num,
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
                                                     layer_depth=self.layer_depth - 1,
                                                     )

    def call(self, inputs, **kwargs):
        att2 = self.attention_layer2(inputs)
        feed2 = self.feed_forward_layer2(att2)
        return feed2


class DecoderComponent(Component):
    def __init__(self,
                 head_num,
                 hidden_dim,
                 name='decode',
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',
                 input_shape=None,
                 *args,
                 **kwargs):
        super(DecoderComponent, self).__init__(name=name)
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation
        self.supports_masking = True

        self.self_attention_layer = self.query_attention_layer = self.feed_forward_layer = None
        super(DecoderComponent, self).__init__(name=name, *args, **kwargs)

    def build(self, input_shape):
        self.self_attention_layer = WrapCodeComponent(head_num=self.head_num,
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
                                                      layer_depth=self.layer_depth - 1,
                                                      )

        self.query_attention_layer = WrapCodeComponent(head_num=self.head_num,
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
                                                       layer_depth=self.layer_depth - 1,
                                                       )

        self.feed_forward_layer = WrapCodeComponent(head_num=self.head_num,
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
                                                    layer_depth=self.layer_depth - 1,
                                                    )
        pass

    def call(self, inputs, **kwargs):
        input_layer, encoded_layer = inputs

        self_att2 = self.self_attention_layer(input_layer)
        query_att2 = self.query_attention_layer(
            [self_att2, encoded_layer, encoded_layer])
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


class EncoderList(Component):
    def __init__(self, encoder_num, name='encoder', layer_depth=1, **kwargs):
        super(EncoderList, self).__init__(name=name)
        self.encoder_num = encoder_num

        self.encodes = []
        self.kwargs = kwargs

        super(EncoderList, self).__init__(name=name, layer_depth=layer_depth)

    def build(self, input_shape):
        for i in range(self.encoder_num):
            self.encodes.append(EncoderComponent(name='{}_{}'.format(self.name, i + 1),
                                                 layer_depth=self.layer_depth - 1, **self.kwargs))

    def call(self, inputs, **kwargs):
        last_layer = inputs
        for layer in self.encodes:
            last_layer = layer(last_layer)
        return last_layer


class DecoderList(Component):
    def __init__(self, decoder_num, name='decoder', layer_depth=1, **kwargs):
        super(DecoderList, self).__init__(name=name)
        self.decoder_num = decoder_num
        self.decodes = []
        self.kwargs = kwargs
        super(DecoderList, self).__init__(name=name, layer_depth=layer_depth)

    def build(self, input_shape):
        for i in range(self.decoder_num):
            self.decodes.append(DecoderComponent(name='{}_{}'.format(self.name, i + 1),
                                                 layer_depth=self.layer_depth - 1,
                                                 **self.kwargs))

    def call(self, inputs, **kwargs):
        last_layer, encoded_layer = inputs
        for layer in self.decodes:
            last_layer = layer([last_layer, encoded_layer])
        return last_layer
