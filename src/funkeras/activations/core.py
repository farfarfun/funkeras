import math
from typing import Any

from tensorflow.keras import backend as K
from tensorflow.python.ops.math_ops import erf, sqrt


def gelu(x: Any) -> Any:
    """GELU 激活函数的精确实现（基于误差函数 erf）。

    参数:
        x: 输入张量。

    返回:
        与输入同形状的激活结果张量。
    """
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def gelu2(x: Any) -> Any:
    """GELU 激活函数的 tanh 近似实现（计算更快，精度略低于 :func:`gelu`）。

    参数:
        x: 输入张量。

    返回:
        与输入同形状的激活结果张量。
    """
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
