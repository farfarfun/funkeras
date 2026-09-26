from farlog import getLogger
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

logger = getLogger("funkeras")


def textcnn(
    max_sequence_length: int,
    max_token_num: int,
    embedding_dim: int,
    output_dim: int,
    model_img_path: str | None = None,
    embedding_matrix=None,
) -> Model:
    """构建 TextCNN 文本分类模型。

    结构：1. embedding 层，2. 多个卷积核尺寸的卷积层，3. max-pooling，4. softmax 全连接层。

    Args:
        max_sequence_length: 输入序列的最大长度（padding 后的 token 数）。
        max_token_num: 词表大小，即 embedding 层的输入维度。
        embedding_dim: 词向量维度。
        output_dim: 分类类别数，即输出层维度。
        model_img_path: 若提供，则把模型结构图保存到该路径；为 None 时不画图。
        embedding_matrix: 预训练词向量矩阵，形状为 (max_token_num, embedding_dim)；
            为 None 时使用随机初始化的可训练 Embedding。

    Returns:
        构建好的 Keras `Model`，输入为形状 `(max_sequence_length,)` 的 token id 序列，
        输出为形状 `(output_dim,)` 的 softmax 概率分布。
    """
    x_input = Input(shape=(max_sequence_length,))
    logger.debug(f"x_input.shape: {x_input.shape}")  # (?, 60)

    if embedding_matrix is None:
        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length)(x_input)
    else:
        x_emb = Embedding(input_dim=max_token_num, output_dim=embedding_dim, input_length=max_sequence_length,
                          weights=[embedding_matrix], trainable=True)(x_input)
    logger.debug(f"x_emb.shape: {x_emb.shape}")  # (?, 60, 300)

    pool_output = []
    kernel_sizes = [2, 3, 4]
    for kernel_size in kernel_sizes:
        c = Conv1D(filters=2, kernel_size=kernel_size, strides=1)(x_emb)
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
        logger.debug(f"kernel_size: {kernel_size} \t c.shape: {c.shape} \t p.shape: {p.shape}")
    pool_output = concatenate([p for p in pool_output])
    logger.debug(f"pool_output.shape: {pool_output.shape}")  # (?, 1, 6)

    x_flatten = Flatten()(pool_output)  # (?, 6)
    y = Dense(output_dim, activation='softmax')(x_flatten)  # (?, 2)
    logger.debug(f"y.shape: {y.shape}")

    model = Model([x_input], outputs=[y])
    if model_img_path:
        plot_model(model, to_file=model_img_path, show_shapes=True, show_layer_names=False)
    model.summary()
    return model
