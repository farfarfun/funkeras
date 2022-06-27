import math

import numpy as np
from tensorflow.keras.initializers import Initializer


class PriorProbability(Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        if type(dtype) is not str:
            dtype = 'float32'
        result = np.ones(shape, dtype=dtype) * - \
            math.log((1 - self.probability) / self.probability)

        return result
