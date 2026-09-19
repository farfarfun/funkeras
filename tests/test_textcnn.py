"""funkeras.models.text.textcnn 的正常路径测试。

textcnn 依赖 tensorflow，用 importorskip 在没有装 tensorflow 的环境下优雅跳过。
"""

import pytest

tf = pytest.importorskip("tensorflow", reason="textcnn 基于 tensorflow.keras 构建")


def test_textcnn_builds_model_with_expected_input_output_shape():
    from funkeras.models.text import textcnn

    model = textcnn(
        max_sequence_length=10,
        max_token_num=100,
        embedding_dim=8,
        output_dim=3,
    )

    assert model.input_shape == (None, 10)
    assert model.output_shape == (None, 3)


def test_textcnn_accepts_pretrained_embedding_matrix():
    import numpy as np

    from funkeras.models.text import textcnn

    embedding_matrix = np.random.rand(50, 8).astype("float32")
    model = textcnn(
        max_sequence_length=6,
        max_token_num=50,
        embedding_dim=8,
        output_dim=2,
        embedding_matrix=embedding_matrix,
    )

    assert model.output_shape == (None, 2)
