"""funkeras.ops.ops.autodetect_mode 的正常路径与边界测试。

`ops.py` 基于 tensorflow 构建张量秩判断逻辑，在没有安装 tensorflow 的环境下
用 importorskip 优雅跳过，而不是让 `tests/` 整体失败。
"""

import pytest

tf = pytest.importorskip("tensorflow", reason="autodetect_mode 依赖 tensorflow.keras.backend")


def test_autodetect_mode_single_when_both_rank_2():
    """x、a 均为 2 阶张量（单图模式）时应返回 SINGLE。"""
    from funkeras.ops import ops

    x = tf.zeros((4, 8))
    a = tf.zeros((4, 4))

    assert ops.autodetect_mode(x, a) == ops.SINGLE


def test_autodetect_mode_batch_when_both_rank_3():
    """x、a 均为 3 阶张量（批模式）时应返回 BATCH。"""
    from funkeras.ops import ops

    x = tf.zeros((2, 4, 8))
    a = tf.zeros((2, 4, 4))

    assert ops.autodetect_mode(x, a) == ops.BATCH


def test_autodetect_mode_mixed_when_x_rank_3_a_rank_2():
    """x 为 3 阶、a 为 2 阶（混合模式，邻接矩阵跨批共享）时应返回 MIXED。"""
    from funkeras.ops import ops

    x = tf.zeros((2, 4, 8))
    a = tf.zeros((4, 4))

    assert ops.autodetect_mode(x, a) == ops.MIXED


def test_autodetect_mode_unsupported_rank_combo_raises_value_error():
    """x 为 2 阶、a 为 3 阶不属于任何已支持组合，应抛出 ValueError 而不是静默出错。"""
    from funkeras.ops import ops

    x = tf.zeros((4, 8))
    a = tf.zeros((2, 4, 4))

    with pytest.raises(ValueError):
        ops.autodetect_mode(x, a)
