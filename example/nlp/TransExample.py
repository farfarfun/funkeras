import numpy as np

from notekeras.backend import plot_model
from notekeras.model.transformer import TransformerModel
from notekeras.tokenizer import get_base_dict, TOKEN_PAD, TOKEN_END, TOKEN_START

tokens = 'all work and no play makes jack a dull boy'.split(' ')
token_dict = get_base_dict(tokens)

# 生成toy数据
encoder_inputs_no_padding = []
encoder_inputs, decoder_inputs, decoder_outputs = [], [], []
for i in range(1, len(tokens) - 1):
    encode_tokens, decode_tokens = tokens[:i], tokens[i:]
    encode_tokens = [TOKEN_START] + encode_tokens + [TOKEN_END] + [TOKEN_PAD] * (len(tokens) - len(encode_tokens))
    output_tokens = decode_tokens + [TOKEN_END, TOKEN_PAD] + [TOKEN_PAD] * (len(tokens) - len(decode_tokens))
    decode_tokens = [TOKEN_START] + decode_tokens + [TOKEN_END] + [TOKEN_PAD] * (len(tokens) - len(decode_tokens))
    encode_tokens = list(map(lambda x: token_dict[x], encode_tokens))
    decode_tokens = list(map(lambda x: token_dict[x], decode_tokens))
    output_tokens = list(map(lambda x: [token_dict[x]], output_tokens))
    encoder_inputs_no_padding.append(encode_tokens[:i + 2])
    encoder_inputs.append(encode_tokens)
    decoder_inputs.append(decode_tokens)
    decoder_outputs.append(output_tokens)

# 构建模型

model = TransformerModel(token_num=len(token_dict),
                         embed_dim=30,
                         encoder_num=3,
                         decoder_num=2,
                         head_num=3,
                         hidden_dim=120,
                         attention_activation='relu',
                         feed_forward_activation='relu',
                         dropout_rate=0.05,
                         name='trans',
                         embed_weights=np.random.random((len(token_dict), 30)),
                         layer_depth=5
                         )
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
)
model.summary()
plot_model(model, to_file='transform.png', show_shapes=True)
plot_model(model, to_file='transform-expand.png', show_shapes=True, expand_nested=True)

# Train the model
model.fit(
    x=[np.asarray(encoder_inputs * 1000), np.asarray(decoder_inputs * 1000)],
    y=np.asarray(decoder_outputs * 1000),
    epochs=1,
)

decoded = model.decode(
    encoder_inputs_no_padding,
    start_token=token_dict[TOKEN_START],
    end_token=token_dict[TOKEN_END],
    pad_token=token_dict[TOKEN_PAD],
    max_len=100
)

token_dict_rev = {v: k for k, v in token_dict.items()}
for i in range(len(decoded)):
    print(' '.join(map(lambda x: token_dict_rev[x], decoded[i][1:-1])))
