"""兼容层：`import notekeras` 已废弃，请改用 `import funkeras`。

这个模块只做一件事：把 `notekeras` 转发到 `funkeras`，保证已经在用
`import notekeras` / `from notekeras...` 的代码在升级后不会立刻报错。
计划在 1.0.0 移除这个兼容层，详见 CHANGELOG.md。
"""

import sys
import warnings

import funkeras

warnings.warn(
    "`import notekeras` 已废弃，请改用 `import funkeras`。"
    "这个兼容层计划在 1.0.0 移除。",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = funkeras
