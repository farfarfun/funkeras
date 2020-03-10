import numpy as np
from tensorflow import keras

from notekeras.backend import plot_model
from notekeras.component.transformer import EncoderModel2, encode_list, WrapCodeModel3
from notekeras.layer import MultiHeadAttention, ScaledDotProductAttention


def data_mock(batch=32, size=None, feature=10):
    if size is None:
        size = [3, 5, 5]
    key = np.random.rand(batch, size[0], feature)

    query = np.random.rand(batch, size[1], feature)
    value = np.random.rand(batch, size[2], feature)
    return key, query, value


def attention_example():
    layer = ScaledDotProductAttention()
    key, query, value = data_mock()

    print("self attention")
    print(layer(key))
    print("attention")
    print(layer([key, query, value]))


def multi_attention_example():
    layer = MultiHeadAttention(2)
    key, query, value = data_mock()

    print("self-multi attention")
    print(layer(key))
    print("multi attention")
    print(layer([key, query, value]))


def wrap_attention_example():
    key, query, value = data_mock()

    # layer = WrapCodeModel2(name='wrap', head_num=2, hidden_dim=2, as_layer=True, input_shape=np.shape(key))
    input = keras.layers.Input(shape=np.shape(key), name='Encoder-Input')
    layer = WrapCodeModel3(name='aaa', head_num=2,
                           hidden_dim=2,
                           use_attention=False,
                           input_shape=np.shape(key))
    output = layer(input)
    output = keras.layers.Dense(4)(output)
    model = keras.models.Model(input, output)
    plot_model(model, to_file='wrap_attention_example.png', show_shapes=True)
    plot_model(model, to_file='wrap_attention_example-expand.png', show_shapes=True, expand_nested=True)

    print("self-multi attention")
    print(model(key))
    print("multi attention")
    print(model([key, query, value]))


def encode_example():
    key, query, value = data_mock()

    input = keras.layers.Input(shape=np.shape(key), name='Encoder_Input')
    layer1 = EncoderModel2(name='wrap', head_num=2, hidden_dim=2, input_shape=np.shape(key))
    input1 = layer1(input)

    output = keras.layers.Dense(3)(input1)
    model = keras.models.Model(input, output)

    # model = EncoderModel3(name='wrap', head_num=2, hidden_dim=2, input_shape=np.shape(key))
    plot_model(model, to_file='encode.png', show_shapes=True)
    plot_model(model, to_file='encode-expand.png', show_shapes=True, expand_nested=True)

    print("self-multi attention")
    print(model(key))
    print("multi attention")
    print(model([key, query, value]))


def encode_list_example():
    key, query, value = data_mock()

    input = keras.layers.Input(shape=np.shape(key), name='Encoder-Input')
    print(np.shape(key))
    print(np.shape(input)[1:])

    output1 = EncoderModel2(name='wrap', head_num=2, hidden_dim=2, as_layer=True, input_shape=np.shape(key))(input)
    output2 = EncoderModel2(name='wrap_2', head_num=2, hidden_dim=2, as_layer=True, input_shape=np.shape(key))(output1)
    output3 = EncoderModel2(name='wrap3', head_num=2, hidden_dim=2, as_layer=True, input_shape=np.shape(key))(output2)
    output4 = EncoderModel2(name='wrap4', head_num=2, hidden_dim=2, as_layer=True, input_shape=np.shape(key))(output3)
    output = EncoderModel2(name='wrap5', head_num=2, hidden_dim=2, as_layer=True, input_shape=np.shape(key))(output4)

    output = encode_list(inputs=output, encoder_num=2, head_num=2, hidden_dim=2)

    model = keras.models.Model(input, output)

    plot_model(model, to_file='encode-list.png', show_shapes=True)
    plot_model(model, to_file='encode-list-expand.png', show_shapes=True, expand_nested=True)

    print("self-multi attention")
    print(model(key))
    print("multi attention")
    print(model([key, query, value]))


# attention_example()
# multi_attention_example()
# wrap_attention_example()
encode_example()
# encode_list_example()
