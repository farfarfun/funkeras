"""
模型组件，准确来说不是一个Layer，也不是一个Model，只是一些Layer的组合

"""

from .components import WideDeepComponent
from .core import Component
from .transformer import EncoderLayer, DecoderLayer
from .transformer import EncoderList, DecoderList
