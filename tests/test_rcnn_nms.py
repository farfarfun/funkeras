"""funkeras.models.rcnn.utils.nms 中回归函数的正常路径与边界测试。

这两个函数只依赖 numpy/math，但 `nms` 模块通过
`iou -> image_processing -> cv2` 这条导入链间接依赖 opencv-python，
所以用 importorskip 在没有装 cv2 的环境下优雅跳过，而不是让 `tests/` 整体失败。
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="funkeras.models.rcnn 通过 image_processing 依赖 opencv-python")
nms = pytest.importorskip("funkeras.models.rcnn.utils.nms")


def test_apply_regr_identity_when_regression_is_zero():
    """回归量全为 0 时，apply_regr 应该原样返回输入框（正常路径）。"""
    x1, y1, w1, h1 = nms.apply_regr(10, 10, 20, 20, tx=0, ty=0, tw=0, th=0)
    assert (x1, y1, w1, h1) == (10, 10, 20, 20)


def test_apply_regr_overflow_falls_back_to_original_box():
    """tw/th 大到 math.exp 溢出时，应捕获 OverflowError 并回退到原始框，而不是抛异常。"""
    x1, y1, w1, h1 = nms.apply_regr(10, 10, 20, 20, tx=0, ty=0, tw=10000, th=0)
    assert (x1, y1, w1, h1) == (10, 10, 20, 20)


def test_apply_regr_np_identity_when_regression_is_zero():
    """apply_regr_np 是 apply_regr 的向量化版本，零回归量时同样应保持坐标不变。"""
    x = np.full((2, 2), 10.0)
    y = np.full((2, 2), 10.0)
    w = np.full((2, 2), 20.0)
    h = np.full((2, 2), 20.0)
    X = np.stack([x, y, w, h])
    T = np.zeros_like(X)

    result = nms.apply_regr_np(X, T)

    assert np.allclose(result, X)


def test_apply_regr_np_invalid_shape_falls_back_to_input():
    """输入形状不满足 X[0:4, :, :] 的前提时，应捕获异常并原样返回 X，而不是抛出未处理异常。"""
    X = np.zeros((3, 2, 2))  # 少了一个通道，X[3, :, :] 会越界
    T = np.zeros((3, 2, 2))

    result = nms.apply_regr_np(X, T)

    assert result is X
