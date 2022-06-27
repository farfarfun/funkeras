import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from notekeras.component.transformer import EncoderList, DecoderList
from notekeras.layers.attention import MultiHeadAttention
from notekeras.layers.embedding import EmbeddingRet, EmbeddingSim
from notekeras.layers.embedding import TrigPosEmbedding
from notekeras.layers.feed_forward import FeedForward
from notekeras.layers.normalize import LayerNormalization

__all__ = ['get_custom_objects', 'TransformerModel']


def get_custom_objects():
    return {
        'LayerNormalization': LayerNormalization,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForward': FeedForward,
        'TrigPosEmbedding': TrigPosEmbedding,
        'EmbeddingRet': EmbeddingRet,
        'EmbeddingSim': EmbeddingSim,
    }


class TransformerModel(Model):
    def __init__(self,
                 token_num,
                 embed_dim,
                 encoder_num,
                 decoder_num,
                 head_num,
                 hidden_dim,
                 name='transformer',
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 use_same_embed=True,
                 embed_weights=None,
                 embed_trainable=None,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',
                 layer_depth=2
                 ):
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

        self.token_num = token_num
        self.embed_dim = embed_dim
        self.encoder_num = encoder_num
        self.decoder_num = decoder_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim

        self.attention_activation = attention_activation
        self.feed_forward_activation = feed_forward_activation
        self.dropout_rate = dropout_rate

        self.use_same_embed = use_same_embed
        self.embed_weights = embed_weights
        self.embed_trainable = embed_trainable
        self.trainable = trainable
        self.use_adapter = use_adapter
        self.adapter_units = adapter_units
        self.adapter_activation = adapter_activation

        self.input_layer = self.output_layer = None
        self.layer_depth = layer_depth

        self._build()

        super(TransformerModel, self).__init__(inputs=self.input_layer, outputs=self.output_layer, name=name,
                                               trainable=trainable)

    def _build(self):
        if not isinstance(self.token_num, list):
            self.token_num = [self.token_num, self.token_num]
        encoder_token_num, decoder_token_num = self.token_num

        if not isinstance(self.embed_weights, list):
            self.embed_weights = [self.embed_weights, self.embed_weights]
        encoder_embed_weights, decoder_embed_weights = self.embed_weights

        if not isinstance(self.embed_trainable, list):
            self.embed_trainable = [self.embed_trainable, self.embed_trainable]
        encoder_embed_trainable, decoder_embed_trainable = self.embed_trainable

        if encoder_embed_weights is not None:
            encoder_embed_weights = [encoder_embed_weights]
        if decoder_embed_weights is not None:
            decoder_embed_weights = [decoder_embed_weights]

        if encoder_embed_trainable is None:
            encoder_embed_trainable = encoder_embed_weights is None
        if decoder_embed_trainable is None:
            decoder_embed_trainable = decoder_embed_weights is None

        if self.use_same_embed:
            encoder_embed_layer = decoder_embed_layer = EmbeddingRet(input_dim=encoder_token_num,
                                                                     output_dim=self.embed_dim,
                                                                     mask_zero=True,
                                                                     weights=encoder_embed_weights,
                                                                     trainable=encoder_embed_trainable,
                                                                     name='Token-Embedding',
                                                                     )
        else:
            encoder_embed_layer = EmbeddingRet(input_dim=encoder_token_num,
                                               output_dim=self.embed_dim,
                                               mask_zero=True,
                                               weights=encoder_embed_weights,
                                               trainable=encoder_embed_trainable,
                                               name='Encoder-Token-Embedding',
                                               )
            decoder_embed_layer = EmbeddingRet(input_dim=decoder_token_num,
                                               output_dim=self.embed_dim,
                                               mask_zero=True,
                                               weights=decoder_embed_weights,
                                               trainable=decoder_embed_trainable,
                                               name='Decoder-Token-Embedding',
                                               )

        encoder_input = Input(shape=(None,), name='Encoder-Input')
        encoder_embed = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_ADD,
                                         name='Encoder-Embedding',
                                         )(encoder_embed_layer(encoder_input)[0])

        encoded_layer = EncoderList(encoder_num=self.encoder_num,
                                    head_num=self.head_num,
                                    hidden_dim=self.hidden_dim,
                                    attention_activation=self.attention_activation,
                                    feed_forward_activation=self.feed_forward_activation,
                                    dropout_rate=self.dropout_rate,
                                    trainable=self.trainable,
                                    use_adapter=self.use_adapter,
                                    adapter_units=self.adapter_units,
                                    adapter_activation=self.adapter_activation,
                                    input_shape=np.shape(encoder_embed)[1:],
                                    layer_depth=self.layer_depth
                                    )(encoder_embed)

        decoder_input = Input(shape=(None,), name='Decoder-Input')
        decoder_embed, decoder_embed_weights = decoder_embed_layer(decoder_input)
        decoder_embed = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_ADD, name='Decoder-Embedding', )(decoder_embed)

        decoded_layer = DecoderList(decoder_num=self.decoder_num,
                                    head_num=self.head_num,
                                    hidden_dim=self.hidden_dim,
                                    attention_activation=self.attention_activation,
                                    feed_forward_activation=self.feed_forward_activation,
                                    dropout_rate=self.dropout_rate,
                                    trainable=self.trainable,
                                    use_adapter=self.use_adapter,
                                    adapter_units=self.adapter_units,
                                    adapter_activation=self.adapter_activation,
                                    input_shape=np.shape(decoder_embed)[1:],
                                    layer_depth=self.layer_depth
                                    )([decoder_embed, encoded_layer])

        dense_layer = EmbeddingSim(trainable=self.trainable, name='Output', )([decoded_layer, decoder_embed_weights])

        self.input_layer = [encoder_input, decoder_input]
        self.output_layer = dense_layer

        pass

    def decode(self,
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
            predicts = self.predict([np.array(batch_inputs), np.array(batch_outputs)])

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
                        self._get_max_suffix_repeat_times(decoder_inputs, max_repeat * max_repeat_block) >= max_repeat:
                    outputs[index_map[i]] = decoder_inputs[index_map[i]]
        if is_single:
            outputs = outputs[0]
        return outputs

    @staticmethod
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
