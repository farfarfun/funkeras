import math

from tensorflow.keras import backend as K
from tensorflow.python.ops.math_ops import erf, sqrt


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def gelu2(x):
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
