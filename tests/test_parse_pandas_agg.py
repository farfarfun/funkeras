"""funkeras.features.parse_pandas 的 agg_set/agg_list 正常路径与边界测试。

`parse_pandas` 模块顶层 `import tensorflow as tf`，用 importorskip 在没有装
tensorflow 的环境下优雅跳过。
"""

import pytest

pytest.importorskip("tensorflow", reason="parse_pandas 顶层导入 tensorflow")


def test_agg_set_without_limit_returns_all_unique_values():
    """size <= 0 时不限制长度，只去重。"""
    from funkeras.features.parse_pandas import agg_set

    agg = agg_set()
    result = agg([1, 1, 2, 3, 3])

    assert set(result) == {1, 2, 3}
    assert len(result) == 3


def test_agg_set_with_limit_truncates_without_padding():
    """size > 0 且 padding=False 时，截断到 size 长度，不做补齐。"""
    from funkeras.features.parse_pandas import agg_set

    agg = agg_set(size=2, padding=False)
    result = agg([1, 2, 3, 4])

    assert len(result) == 2


def test_agg_set_pads_short_input_with_zero_string():
    """边界：去重后长度不足 size 时，用字符串 '0' 补齐到指定长度。"""
    from funkeras.features.parse_pandas import agg_set

    agg = agg_set(size=5, padding=True)
    result = agg([1, 2])

    assert len(result) == 5
    assert result.count('0') == 3


def test_agg_list_without_limit_preserves_order_and_duplicates():
    """agg_list 不去重：size <= 0 时原样转换为 list，保留重复值与顺序。"""
    from funkeras.features.parse_pandas import agg_list

    agg = agg_list()
    assert agg([3, 1, 1, 2]) == [3, 1, 1, 2]


def test_agg_list_pads_short_input_with_zero_string():
    """边界：长度不足 size 时用字符串 '0' 补齐。"""
    from funkeras.features.parse_pandas import agg_list

    agg = agg_list(size=4, padding=True)
    result = agg([1, 2])

    assert result == [1, 2, '0', '0']
