import numpy as np

from notekeras.backend import keras
from notekeras.layer.attention import MultiHeadAttention
from notekeras.layer.embedding import EmbeddingRet, EmbeddingSim
from notekeras.layer.embedding import TrigPosEmbedding
from notekeras.layer.feed_forward import FeedForward
from notekeras.layer.normalize import LayerNormalization

__all__ = [
    'get_custom_objects', 'get_encoders_layers', 'get_decoders_layers', 'get_model', 'decode',
    'attention_builder', 'feed_forward_builder', 'get_encoder_component', 'get_decoder_component',
]


def get_custom_objects():
    return {
        'LayerNormalization': LayerNormalization,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForward': FeedForward,
        'TrigPosEmbedding': TrigPosEmbedding,
        'EmbeddingRet': EmbeddingRet,
        'EmbeddingSim': EmbeddingSim,
    }


class WrapLayer(keras.layers.Layer):
    def __init__(self,
                 name,
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',
                 is_list=False,
                 ):
        super(WrapLayer, self).__init__()
        self.name = name
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation

        self.dropout_layer1 = self.dropout_layer2 = self.adapter = None
        self.add_layer = self.adapter = self.normal_layer = None
        self.is_list = is_list

        self.supports_masking = True
        self.layers = [self.dropout_layer1, self.dropout_layer2, self.adapter, self.add_layer, self.normal_layer]

    def build(self, input_shape):
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
        self.layers = [self.dropout_layer1, self.dropout_layer2, self.adapter, self.add_layer, self.normal_layer]
        super(WrapLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        print((self.name, input_shape))
        if self.is_list and False:
            return input_shape[1:]
        return input_shape

    def call(self, input_layer, **kwargs):
        build_output = input_layer

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
            'use_adapter': self.use_adapter,
            'adapter_units': self.adapter_units,
            'adapter_activation': self.adapter_activation,
        }
        for layer in self.layers:
            if layer is not None:
                config['layers'].append({
                    'class_name': layer.__class__.__name__,
                    'config': layer.get_config(),
                })
        base_config = super(WrapLayer, self).get_config()
        # base_config.pop('layer')
        # return dict(list(base_config.items()) + list(config.items()))
        return dict(base_config, **config)


class EncoderComponent(keras.layers.Layer):
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

        self.attention_layer1 = self.attention_layer2 = None
        self.feed_forward_layer1 = self.feed_forward_layer2 = None

        self.supports_masking = True

        self.attention_layer1 = MultiHeadAttention(name='%s-MultiHeadSelfAttention' % self.name,
                                                   head_num=self.head_num,
                                                   activation=self.attention_activation,
                                                   history_only=False,
                                                   trainable=self.trainable,
                                                   )

        self.attention_layer2 = WrapLayer(name='%s-MultiHeadSelfAttention-WrapLayer' % self.name,
                                          dropout_rate=self.dropout_rate,
                                          trainable=self.trainable,
                                          use_adapter=self.use_adapter,
                                          adapter_units=self.adapter_units,
                                          adapter_activation=self.adapter_activation,
                                          )

        self.feed_forward_layer1 = FeedForward(name='%s-FeedForward' % self.name,
                                               units=self.hidden_dim,
                                               activation=self.feed_forward_activation,
                                               trainable=self.trainable,
                                               )

        self.feed_forward_layer2 = WrapLayer(name='%s-FeedForward-WrapLayer' % self.name,
                                             dropout_rate=self.dropout_rate,
                                             trainable=self.trainable,
                                             use_adapter=self.use_adapter,
                                             adapter_units=self.adapter_units,
                                             adapter_activation=self.adapter_activation,
                                             )

        self.layers = [self.attention_layer1, self.attention_layer2, self.feed_forward_layer1, self.feed_forward_layer2]

    def build(self, input_shape):
        super(EncoderComponent, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        att1 = self.attention_layer1(inputs)
        att2 = self.attention_layer2(att1)
        feed1 = self.feed_forward_layer1(att2)
        feed2 = self.feed_forward_layer2(feed1)

        return feed2

    def get_config(self):

        config = {
            'layers': [],
            'dropout_rate': self.dropout_rate,
            'use_adapter': self.use_adapter,
            'adapter_units': self.adapter_units,
            'adapter_activation': self.adapter_activation,
        }
        for layer in self.layers:
            if layer is not None:
                config['layers'].append({
                    'class_name': layer.__class__.__name__,
                    'config': layer.get_config(),
                })
        base_config = super(EncoderComponent, self).get_config()
        # base_config.pop('layer')
        # return dict(list(base_config.items()) + list(config.items()))
        return dict(base_config, **config)


class DecoderComponent(keras.layers.Layer):
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

        self.self_attention_layer1 = self.self_attention_layer2 = None
        self.feed_forward_layer1 = self.feed_forward_layer2 = None
        self.query_attention_layer1 = self.query_attention_layer2 = None
        self.supports_masking = True

    def build(self, input_shape):
        self_attention_name = '%s-MultiHeadSelfAttention' % self.name
        query_attention_name = '%s-MultiHeadQueryAttention' % self.name
        feed_forward_name = '%s-FeedForward' % self.name

        self.self_attention_layer1 = MultiHeadAttention(name=self_attention_name,
                                                        head_num=self.head_num,
                                                        activation=self.attention_activation,
                                                        history_only=True,
                                                        trainable=self.trainable,
                                                        )

        self.self_attention_layer2 = WrapLayer(name=self_attention_name,
                                               dropout_rate=self.dropout_rate,
                                               trainable=self.trainable,
                                               use_adapter=self.use_adapter,
                                               adapter_units=self.adapter_units,
                                               adapter_activation=self.adapter_activation,
                                               )
        self.query_attention_layer1 = MultiHeadAttention(name=query_attention_name,
                                                         head_num=self.head_num,
                                                         activation=self.attention_activation,
                                                         history_only=False,
                                                         trainable=self.trainable,
                                                         )

        self.query_attention_layer2 = WrapLayer(name=query_attention_name,
                                                dropout_rate=self.dropout_rate,
                                                trainable=self.trainable,
                                                use_adapter=self.use_adapter,
                                                adapter_units=self.adapter_units,
                                                adapter_activation=self.adapter_activation,
                                                is_list=True
                                                )

        self.feed_forward_layer1 = FeedForward(name=feed_forward_name,
                                               units=self.hidden_dim,
                                               activation=self.feed_forward_activation,
                                               trainable=self.trainable,
                                               )

        self.feed_forward_layer2 = WrapLayer(name=feed_forward_name,
                                             dropout_rate=self.dropout_rate,
                                             trainable=self.trainable,
                                             use_adapter=self.use_adapter,
                                             adapter_units=self.adapter_units,
                                             adapter_activation=self.adapter_activation,
                                             )
        # print(self.feed_forward_layer2.compute_output_shape(input_shape))
        super(DecoderComponent, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        input_layer, encoded_layer = inputs

        self_att1 = self.self_attention_layer1(input_layer)
        self_att2 = self.self_attention_layer2(self_att1)

        query_att1 = self.query_attention_layer1([self_att2, encoded_layer, encoded_layer])
        query_att2 = self.query_attention_layer2(query_att1)

        feed1 = self.feed_forward_layer1(query_att2)
        feed2 = self.feed_forward_layer2(feed1)

        return feed2


def _wrap_layer(name,
                input_layer,
                build_func,
                dropout_rate=0.0,
                trainable=True,
                use_adapter=False,
                adapter_units=None,
                adapter_activation='relu'):
    """ 包装 Wrap layers with residual, normalization and dropout.

    :param name: 名称的前缀 Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param build_func: A callable that takes the input tensor and generates the output tensor.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Output layer.
    """
    build_output = build_func(input_layer)

    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(rate=dropout_rate, name='%s-Dropout' % name, )(build_output)
    else:
        dropout_layer = build_output
    if isinstance(input_layer, list):
        input_layer = input_layer[0]
    if use_adapter:
        adapter = FeedForward(units=adapter_units,
                              activation=adapter_activation,
                              kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
                              name='%s-Adapter' % name,
                              )(dropout_layer)
        dropout_layer = keras.layers.Add(name='%s-Adapter-Add' % name)([dropout_layer, adapter])
    #
    add_layer = keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])

    # 正则化
    normal_layer = LayerNormalization(trainable=trainable, name='%s-Norm' % name, )(add_layer)
    return normal_layer


def attention_builder(name, head_num, activation, history_only, trainable=True):
    """Get multi-head self-attention builder.

    :param name: Prefix of names for internal layers.
    :param head_num: Number of heads in multi-head self-attention.
    :param activation: Activation for multi-head self-attention.
    :param history_only: Only use history data.
    :param trainable: Whether the layer is trainable.
    :return:
    """

    def _attention_builder(x):
        return MultiHeadAttention(head_num=head_num,
                                  activation=activation,
                                  history_only=history_only,
                                  trainable=trainable,
                                  name=name, )(x)

    return _attention_builder


def feed_forward_builder(name, hidden_dim, activation, trainable=True):
    """Get position-wise feed-forward layer builder.

    :param name: Prefix of names for internal layers.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param activation: Activation for feed-forward layer.
    :param trainable: Whether the layer is trainable.
    :return:
    """

    def _feed_forward_builder(x):
        return FeedForward(units=hidden_dim,
                           activation=activation,
                           trainable=trainable,
                           name=name,
                           )(x)

    return _feed_forward_builder


def get_encoder_component(name,
                          input_layer,
                          head_num,
                          hidden_dim,
                          attention_activation=None,
                          feed_forward_activation='relu',
                          dropout_rate=0.0,
                          trainable=True,
                          use_adapter=False,
                          adapter_units=None,
                          adapter_activation='relu'):
    """Multi-head self-attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
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
    attention_name = '%s-MultiHeadSelfAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_layer(name=attention_name,
                                  input_layer=input_layer,
                                  build_func=attention_builder(name=attention_name,
                                                               head_num=head_num,
                                                               activation=attention_activation,
                                                               history_only=False,
                                                               trainable=trainable,
                                                               ),
                                  dropout_rate=dropout_rate,
                                  trainable=trainable,
                                  use_adapter=use_adapter,
                                  adapter_units=adapter_units,
                                  adapter_activation=adapter_activation,
                                  )
    feed_forward_layer = _wrap_layer(name=feed_forward_name,
                                     input_layer=attention_layer,
                                     build_func=feed_forward_builder(name=feed_forward_name,
                                                                     hidden_dim=hidden_dim,
                                                                     activation=feed_forward_activation,
                                                                     trainable=trainable,
                                                                     ),
                                     dropout_rate=dropout_rate,
                                     trainable=trainable,
                                     use_adapter=use_adapter,
                                     adapter_units=adapter_units,
                                     adapter_activation=adapter_activation,
                                     )
    return feed_forward_layer


def get_decoder_component(name,
                          input_layer,
                          encoded_layer,
                          head_num,
                          hidden_dim,
                          attention_activation=None,
                          feed_forward_activation='relu',
                          dropout_rate=0.0,
                          trainable=True,
                          use_adapter=False,
                          adapter_units=None,
                          adapter_activation='relu'):
    """Multi-head self-attention, multi-head query attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
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
    self_attention_name = '%s-MultiHeadSelfAttention' % name
    query_attention_name = '%s-MultiHeadQueryAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    self_attention_layer = _wrap_layer(name=self_attention_name,
                                       input_layer=input_layer,
                                       build_func=attention_builder(name=self_attention_name,
                                                                    head_num=head_num,
                                                                    activation=attention_activation,
                                                                    history_only=True,
                                                                    trainable=trainable,
                                                                    ),
                                       dropout_rate=dropout_rate,
                                       trainable=trainable,
                                       use_adapter=use_adapter,
                                       adapter_units=adapter_units,
                                       adapter_activation=adapter_activation,
                                       )
    query_attention_layer = _wrap_layer(name=query_attention_name,
                                        input_layer=[self_attention_layer, encoded_layer, encoded_layer],
                                        build_func=attention_builder(name=query_attention_name,
                                                                     head_num=head_num,
                                                                     activation=attention_activation,
                                                                     history_only=False,
                                                                     trainable=trainable,
                                                                     ),
                                        dropout_rate=dropout_rate,
                                        trainable=trainable,
                                        use_adapter=use_adapter,
                                        adapter_units=adapter_units,
                                        adapter_activation=adapter_activation,
                                        )
    feed_forward_layer = _wrap_layer(name=feed_forward_name,
                                     input_layer=query_attention_layer,
                                     build_func=feed_forward_builder(name=feed_forward_name,
                                                                     hidden_dim=hidden_dim,
                                                                     activation=feed_forward_activation,
                                                                     trainable=trainable,
                                                                     ),
                                     dropout_rate=dropout_rate,
                                     trainable=trainable,
                                     use_adapter=use_adapter,
                                     adapter_units=adapter_units,
                                     adapter_activation=adapter_activation,
                                     )
    return feed_forward_layer


def get_encoders_layers(encoder_num,
                        input_layer,
                        head_num,
                        hidden_dim,
                        attention_activation=None,
                        feed_forward_activation='relu',
                        dropout_rate=0.0,
                        trainable=True,
                        use_adapter=False,
                        adapter_units=None,
                        adapter_activation='relu'):
    """编码层 Get encoders.

    :param encoder_num:编码层数 Number of encoder components.
    :param input_layer:输入层 Input layer.
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
    last_layer = input_layer

    for i in range(encoder_num):
        # last_layer = get_encoder_component(name='Encoder-%d' % (i + 1),
        #                                    input_layer=last_layer,
        #                                    head_num=head_num,
        #                                    hidden_dim=hidden_dim,
        #                                    attention_activation=attention_activation,
        #                                    feed_forward_activation=feed_forward_activation,
        #                                    dropout_rate=dropout_rate,
        #                                    trainable=trainable,
        #                                    use_adapter=use_adapter,
        #                                    adapter_units=adapter_units,
        #                                    adapter_activation=adapter_activation,
        #                                    )

        last_layer = EncoderComponent(name='Encoder-%d' % (i + 1),
                                      head_num=head_num,
                                      hidden_dim=hidden_dim,
                                      attention_activation=attention_activation,
                                      feed_forward_activation=feed_forward_activation,
                                      dropout_rate=dropout_rate,
                                      trainable=trainable,
                                      use_adapter=use_adapter,
                                      adapter_units=adapter_units,
                                      adapter_activation=adapter_activation,
                                      )(last_layer)
    return last_layer


def get_decoders_layers(decoder_num,
                        input_layer,
                        encoded_layer,
                        head_num,
                        hidden_dim,
                        attention_activation=None,
                        feed_forward_activation='relu',
                        dropout_rate=0.0,
                        trainable=True,
                        use_adapter=False,
                        adapter_units=None,
                        adapter_activation='relu'):
    """Get decoders.

    :param decoder_num: Number of decoder components.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
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
    last_layer = input_layer
    for i in range(decoder_num):
        last_layer = get_decoder_component(name='Decoder-%d' % (i + 1),
                                           input_layer=last_layer,
                                           encoded_layer=encoded_layer,
                                           head_num=head_num,
                                           hidden_dim=hidden_dim,
                                           attention_activation=attention_activation,
                                           feed_forward_activation=feed_forward_activation,
                                           dropout_rate=dropout_rate,
                                           trainable=trainable,
                                           use_adapter=use_adapter,
                                           adapter_units=adapter_units,
                                           adapter_activation=adapter_activation,
                                           )
        # last_layer = DecoderComponent(name='Decoder-%d' % (i + 1),
        #                               head_num=head_num,
        #                               hidden_dim=hidden_dim,
        #                               attention_activation=attention_activation,
        #                               feed_forward_activation=feed_forward_activation,
        #                               dropout_rate=dropout_rate,
        #                               trainable=trainable,
        #                               use_adapter=use_adapter,
        #                               adapter_units=adapter_units,
        #                               adapter_activation=adapter_activation,
        #                               )([last_layer, encoded_layer])
    return last_layer


def get_model(token_num,
              embed_dim,
              encoder_num,
              decoder_num,
              head_num,
              hidden_dim,
              attention_activation=None,
              feed_forward_activation='relu',
              dropout_rate=0.0,
              use_same_embed=True,
              embed_weights=None,
              embed_trainable=None,
              trainable=True,
              use_adapter=False,
              adapter_units=None,
              adapter_activation='relu'):
    """Get full model without compilation.

    :param token_num: Number of distinct tokens.
    :param embed_dim: Dimension of token embedding.
    :param encoder_num: Number of encoder components.
    :param decoder_num: Number of decoder components.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param use_same_embed: Whether to use the same token embedding layer. `token_num`, `embed_weights` and
                           `embed_trainable` should be lists of two elements if it is False.
    :param embed_weights: Initial weights of token embedding.
    :param embed_trainable: Whether the token embedding is trainable. It will automatically set to False if the given
                            value is None when embedding weights has been provided.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Keras model.
    """
    if not isinstance(token_num, list):
        token_num = [token_num, token_num]
    encoder_token_num, decoder_token_num = token_num

    if not isinstance(embed_weights, list):
        embed_weights = [embed_weights, embed_weights]
    encoder_embed_weights, decoder_embed_weights = embed_weights
    if encoder_embed_weights is not None:
        encoder_embed_weights = [encoder_embed_weights]
    if decoder_embed_weights is not None:
        decoder_embed_weights = [decoder_embed_weights]

    if not isinstance(embed_trainable, list):
        embed_trainable = [embed_trainable, embed_trainable]
    encoder_embed_trainable, decoder_embed_trainable = embed_trainable
    if encoder_embed_trainable is None:
        encoder_embed_trainable = encoder_embed_weights is None
    if decoder_embed_trainable is None:
        decoder_embed_trainable = decoder_embed_weights is None

    if use_same_embed:
        encoder_embed_layer = decoder_embed_layer = EmbeddingRet(input_dim=encoder_token_num,
                                                                 output_dim=embed_dim,
                                                                 mask_zero=True,
                                                                 weights=encoder_embed_weights,
                                                                 trainable=encoder_embed_trainable,
                                                                 name='Token-Embedding',
                                                                 )
    else:
        encoder_embed_layer = EmbeddingRet(input_dim=encoder_token_num,
                                           output_dim=embed_dim,
                                           mask_zero=True,
                                           weights=encoder_embed_weights,
                                           trainable=encoder_embed_trainable,
                                           name='Encoder-Token-Embedding',
                                           )
        decoder_embed_layer = EmbeddingRet(input_dim=decoder_token_num,
                                           output_dim=embed_dim,
                                           mask_zero=True,
                                           weights=decoder_embed_weights,
                                           trainable=decoder_embed_trainable,
                                           name='Decoder-Token-Embedding',
                                           )

    encoder_input = keras.layers.Input(shape=(None,), name='Encoder-Input')
    encoder_embed = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_ADD,
                                     name='Encoder-Embedding',
                                     )(encoder_embed_layer(encoder_input)[0])
    encoded_layer = get_encoders_layers(encoder_num=encoder_num,
                                        input_layer=encoder_embed,
                                        head_num=head_num,
                                        hidden_dim=hidden_dim,
                                        attention_activation=attention_activation,
                                        feed_forward_activation=feed_forward_activation,
                                        dropout_rate=dropout_rate,
                                        trainable=trainable,
                                        use_adapter=use_adapter,
                                        adapter_units=adapter_units,
                                        adapter_activation=adapter_activation,
                                        )

    decoder_input = keras.layers.Input(shape=(None,), name='Decoder-Input')
    decoder_embed, decoder_embed_weights = decoder_embed_layer(decoder_input)
    decoder_embed = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_ADD, name='Decoder-Embedding', )(decoder_embed)

    decoded_layer = get_decoders_layers(decoder_num=decoder_num,
                                        input_layer=decoder_embed,
                                        encoded_layer=encoded_layer,
                                        head_num=head_num,
                                        hidden_dim=hidden_dim,
                                        attention_activation=attention_activation,
                                        feed_forward_activation=feed_forward_activation,
                                        dropout_rate=dropout_rate,
                                        trainable=trainable,
                                        use_adapter=use_adapter,
                                        adapter_units=adapter_units,
                                        adapter_activation=adapter_activation,
                                        )
    dense_layer = EmbeddingSim(trainable=trainable, name='Output', )([decoded_layer, decoder_embed_weights])

    return keras.models.Model(inputs=[encoder_input, decoder_input], outputs=dense_layer)


def _get_max_suffix_repeat_times(tokens, max_len):
    detect_len = min(max_len, len(tokens))
    next = [-1] * detect_len
    k = -1
    for i in range(1, detect_len):
        while k >= 0 and tokens[len(tokens) - i - 1] != tokens[len(tokens) - k - 2]:
            k = next[k]
        if tokens[len(tokens) - i - 1] == tokens[len(tokens) - k - 2]:
            k += 1
        next[i] = k
    max_repeat = 1
    for i in range(2, detect_len):
        if next[i] >= 0 and (i + 1) % (i - next[i]) == 0:
            max_repeat = max(max_repeat, (i + 1) // (i - next[i]))
    return max_repeat


def decode(model,
           tokens,
           start_token,
           end_token,
           pad_token,
           top_k=1,
           temperature=1.0,
           max_len=10000,
           max_repeat=10,
           max_repeat_block=10):
    """根据给定的模型和输入的token进行解码  Decode with the given model and input tokens.

    :param model: The trained model.
    :param tokens: The input tokens of encoder.
    :param start_token: The token that represents the start of a sentence.
    :param end_token: The token that represents the end of a sentence.
    :param pad_token: The token that represents padding.
    :param top_k: Choose the last token from top K.
    :param temperature: Randomness in boltzmann distribution.
    :param max_len: Maximum length of decoded list.
    :param max_repeat: Maximum number of repeating blocks.
    :param max_repeat_block: Maximum length of the repeating block.
    :return: Decoded tokens.
    """
    is_single = not isinstance(tokens[0], list)
    if is_single:
        tokens = [tokens]
    batch_size = len(tokens)
    decoder_inputs = [[start_token] for _ in range(batch_size)]
    outputs = [None for _ in range(batch_size)]
    output_len = 1
    while len(list(filter(lambda x: x is None, outputs))) > 0:
        output_len += 1
        batch_inputs, batch_outputs = [], []
        max_input_len = 0
        index_map = {}
        for i in range(batch_size):
            if outputs[i] is None:
                index_map[len(batch_inputs)] = i
                batch_inputs.append(tokens[i][:])
                batch_outputs.append(decoder_inputs[i])
                max_input_len = max(max_input_len, len(tokens[i]))
        for i in range(len(batch_inputs)):
            batch_inputs[i] += [pad_token] * (max_input_len - len(batch_inputs[i]))
        predicts = model.predict([np.array(batch_inputs), np.array(batch_outputs)])
        for i in range(len(predicts)):
            if top_k == 1:
                last_token = predicts[i][-1].argmax(axis=-1)
            else:
                probs = [(prob, j) for j, prob in enumerate(predicts[i][-1])]
                probs.sort(reverse=True)
                probs = probs[:top_k]
                indices, probs = list(map(lambda x: x[1], probs)), list(map(lambda x: x[0], probs))
                probs = np.array(probs) / temperature
                probs = probs - np.max(probs)
                probs = np.exp(probs)
                probs = probs / np.sum(probs)
                last_token = np.random.choice(indices, p=probs)
            decoder_inputs[index_map[i]].append(last_token)
            if last_token == end_token or \
                    (max_len is not None and output_len >= max_len) or \
                    _get_max_suffix_repeat_times(decoder_inputs, max_repeat * max_repeat_block) >= max_repeat:
                outputs[index_map[i]] = decoder_inputs[index_map[i]]
    if is_single:
        outputs = outputs[0]
    return outputs
