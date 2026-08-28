"""兼容层：`import notekeras` 已废弃，请改用 `import funkeras`。

这个模块只做一件事：把 `notekeras` 转发到 `funkeras`，保证已经在用
`import notekeras` / `from notekeras...` 的代码在升级后不会立刻报错。
计划在下一次破坏性版本中删除这个兼容层。
"""

import sys
import warnings

import funkeras

warnings.warn(
    "`import notekeras` 已废弃，请改用 `import funkeras`。"
    "这个兼容层会在未来某个版本被移除。",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = funkeras
