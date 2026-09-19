"""funkeras 领域异常定义。

统一在这里定义具体的异常类型，避免库代码里出现 ``raise Exception("...")``
这种无法被调用方精确捕获、也丢失了排查上下文的写法。
"""

from __future__ import annotations


class FunkerasError(Exception):
    """funkeras 所有自定义异常的基类。"""


class FeatureConfigError(FunkerasError):
    """特征配置（`params`/`layer_json` 等）不合法时抛出。

    Args:
        message: 具体错误说明。
        context: 出错时的配置片段，便于定位是哪一份 JSON/字典配置有问题。
    """

    def __init__(self, message: str, context: dict | None = None):
        self.context = context
        if context is not None:
            message = f"{message}（context={context!r}）"
        super().__init__(message)
