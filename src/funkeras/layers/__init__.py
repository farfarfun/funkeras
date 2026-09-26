from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Average as Mean
from tensorflow.keras.layers import Dot, Maximum, Minimum, Multiply, Subtract

from .activations import Dice
from .attention import (MultiHeadAttention, ScaledDotProductAttention,
                        SeqSelfAttention, SeqWeightedAttention)
from .conv import MaskedConv1D
from .core import DNN, CrossLayer, Linear, MaskFlatten, SelfMean, SelfSum
from .embedding import (EmbeddingRet, EmbeddingSim, EmbeddingSimilarity,
                        PositionEmbedding, TaskEmbedding, TokenEmbedding,
                        TrigPosEmbedding)
from .extract import Extract
from .feed_forward import FeedForward
from .fm import FFM, FM, FactorizationMachine
from .inputs import get_inputs
from .masked import Masked
from .normalize import BatchNormalizationFreeze, LayerNormalization
from .pooling import MaskedGlobalMaxPool1D
from .wrappers import WeightNormalization
