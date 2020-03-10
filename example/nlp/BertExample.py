from notekeras.backend import keras
from notekeras.model.bert import get_model, BertModel, gen_batch_inputs
from notekeras.tokenizer import get_base_dict

# 随便的输入样例：
sentence_pairs = [
    [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
    [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
    [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
]

# 构建自定义词典
token_dict = get_base_dict()  # 初始化特殊符号，如`[CLS]`

for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)

token_list = list(token_dict.keys())  # Used for selecting a random word

model = BertModel(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=30,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
)
model.compile_model()
model.summary()


def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )


model.fit_generator(
    generator=_generator(),
    steps_per_epoch=1000,
    epochs=100,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
)

# 使用训练好的模型
inputs, output_layer = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=30,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
    training=False,  # 当`training`是`False`，返回值是输入和输出
    trainable=False,  # 模型是否可训练，默认值和`training`相同
    output_layer_num=4,  # 最后几层的输出将合并在一起作为最终的输出，只有当`training`是`False`有效
)
