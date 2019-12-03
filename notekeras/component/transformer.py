from notekeras.backend import keras
from notekeras.layer.attention import MultiHeadAttention
from notekeras.layer.feed_forward import FeedForward
from notekeras.layer.normalize import LayerNormalization

Layer = keras.layers.Layer

__all__ = ['get_custom_objects', 'WrapCodeComponent',
           'EncoderComponent', 'DecoderComponent', 'EncoderListComponent', 'DecoderListComponent']


def get_custom_objects():
    return {
        'WrapCodeComponent': WrapCodeComponent,
        'EncoderComponent': EncoderComponent,
        'DecoderComponent': DecoderComponent,
        'EncoderListComponent': EncoderListComponent,
        'DecoderListComponent': DecoderListComponent
    }


class WrapCodeComponent(Layer):
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
                 use_attention=True
                 ):
        super(WrapCodeComponent, self).__init__()
        self.name = name
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
        self._build()

    def _build(self):
        if self.use_attention:
            self.attention_layer = MultiHeadAttention(name='%s-Attention' % self.name,
                                                      head_num=self.head_num,
                                                      activation=self.attention_activation,
                                                      history_only=self.history_only,
                                                      trainable=self.trainable,
                                                      )
        else:
            self.attention_layer = FeedForward(name='%s-FeedForward' % self.name,
                                               units=self.hidden_dim,
                                               activation=self.feed_forward_activation,
                                               trainable=self.trainable,
                                               )

        if self.dropout_rate > 0.0:
            self.dropout_layer1 = keras.layers.Dropout(rate=self.dropout_rate, name='%s-Dropout' % self.name, )

        if self.use_adapter:
            self.adapter = FeedForward(units=self.adapter_units,
                                       activation=self.adapter_activation,
                                       kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
                                       name='%s-Adapter' % self.name,
                                       )
            self.dropout_layer2 = keras.layers.Add(name='%s-Adapter-Add' % self.name)
        #
        self.add_layer = keras.layers.Add(name='%s-Add' % self.name)

        # 正则化
        self.normal_layer = LayerNormalization(trainable=self.trainable, name='%s-Norm' % self.name, )

    def __call__(self, input_layer, **kwargs):
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


class EncoderComponent(Layer):
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
                 ):
        super(EncoderComponent, self).__init__()
        self.name = name
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
        self._build()

    def _build(self):
        self.attention_layer2 = WrapCodeComponent(name='%s-MultiHeadSelfAttention' % self.name,
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
                                                  use_attention=True
                                                  )

        self.feed_forward_layer2 = WrapCodeComponent(name='%s-FeedForward' % self.name,
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
                                                     use_attention=False
                                                     )

    def __call__(self, inputs, **kwargs):
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


class DecoderComponent(Layer):
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
                 ):
        super(DecoderComponent, self).__init__()
        self.name = name
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation

        self.self_attention_layer2 = None
        self.feed_forward_layer2 = None
        self.query_attention_layer2 = None
        self.supports_masking = True
        self._build()

    def _build(self):
        self.self_attention_layer2 = WrapCodeComponent(name='%s-SelfAttention' % self.name,
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
                                                       )

        self.query_attention_layer2 = WrapCodeComponent(name='%s-QueryAttention' % self.name,
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
                                                        )

        self.feed_forward_layer2 = WrapCodeComponent(name='%s-FeedForward' % self.name,
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
                                                     )

    def __call__(self, inputs, **kwargs):
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


class EncoderListComponent(Layer):
    def __init__(self,
                 encoder_num,
                 head_num,
                 hidden_dim,
                 name='Encoder',
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu'):
        """编码层 Get encoders.

           :param encoder_num:编码层数 Number of encoder components.
           :param head_num: Number of heads in multi-head self-attention.
           :param hidden_dim: Hidden dimension of feed forward layer.
           :param attention_activation: Activation for multi-head self-attention.
           :param feed_forward_activation: Activation for feed-forward layer.
           :param dropout_rate: Dropout rate.
           :param trainable: Whether the layers are trainable.
           :param use_adapter: Whether to use feed-forward adapters before each residual connections.
           :param adapter_units: The dimension of the first transformation in feed-forward adapter.
           :param adapter_activation: The activation after the first transformation in feed-forward adapter.
           :return: Output layer.
           """
        super(EncoderListComponent, self).__init__()
        self.name = name
        self.encoder_num = encoder_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation

        self.layers = []

        self._build()

    def _build(self):
        for i in range(self.encoder_num):
            self.layers.append(EncoderComponent(name='{}-{}'.format(self.name, i + 1),
                                                head_num=self.head_num,
                                                hidden_dim=self.hidden_dim,
                                                attention_activation=self.attention_activation,
                                                feed_forward_activation=self.feed_forward_activation,
                                                dropout_rate=self.dropout_rate,
                                                trainable=self.trainable,
                                                use_adapter=self.use_adapter,
                                                adapter_units=self.adapter_units,
                                                adapter_activation=self.adapter_activation,
                                                ))
        pass

    def __call__(self, inputs, **kwargs):
        input_layer = inputs
        last_layer = input_layer
        for layer in self.layers:
            last_layer = layer(last_layer)
        return last_layer

    def compute_output_shape(self, input_shape):
        return input_shape


class DecoderListComponent(Layer):
    def __init__(self,
                 decoder_num,
                 head_num,
                 hidden_dim,
                 name='Decoder',
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu'):
        """Get decoders.

            :param decoder_num: Number of decoder components.
            :param head_num: Number of heads in multi-head self-attention.
            :param hidden_dim: Hidden dimension of feed forward layer.
            :param attention_activation: Activation for multi-head self-attention.
            :param feed_forward_activation: Activation for feed-forward layer.
            :param dropout_rate: Dropout rate.
            :param trainable: Whether the layers are trainable.
            :param use_adapter: Whether to use feed-forward adapters before each residual connections.
            :param adapter_units: The dimension of the first transformation in feed-forward adapter.
            :param adapter_activation: The activation after the first transformation in feed-forward adapter.
            :return: Output layer.
            """
        super(DecoderListComponent, self).__init__()
        self.name = name
        self.decoder_num = decoder_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation

        self.layers = []

        self._build()

    def _build(self):
        for i in range(self.decoder_num):
            self.layers.append(DecoderComponent(name='{}-{}'.format(self.name, i + 1),
                                                head_num=self.head_num,
                                                hidden_dim=self.hidden_dim,
                                                attention_activation=self.attention_activation,
                                                feed_forward_activation=self.feed_forward_activation,
                                                dropout_rate=self.dropout_rate,
                                                trainable=self.trainable,
                                                use_adapter=self.use_adapter,
                                                adapter_units=self.adapter_units,
                                                adapter_activation=self.adapter_activation,
                                                ))
        pass

    def __call__(self, inputs, **kwargs):
        input_layer, encoded_layer = inputs
        last_layer = input_layer
        for layer in self.layers:
            last_layer = layer([last_layer, encoded_layer])
        return last_layer

    def compute_output_shape(self, input_shape):
        return input_shape
