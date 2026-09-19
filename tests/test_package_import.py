"""`import funkeras` / `import notekeras` 的正常路径测试。

`funkeras/__init__.py` 本身只声明 __all__、不预先导入任何子模块，所以这两个测试不需要
安装 tensorflow / opencv-python 等重依赖即可运行。
"""

import importlib
import warnings

import funkeras


def test_funkeras_all_lists_top_level_subpackages():
    """__all__ 里声明的子包名必须和实际目录结构一致，避免 `from funkeras import x` 用户踩坑。"""
    expected = {
        "activations", "optimizers", "layers", "models",
        "backend", "component", "features", "ops",
    }
    assert set(funkeras.__all__) == expected


def test_import_notekeras_forwards_to_funkeras_with_deprecation_warning():
    """`notekeras` 兼容层必须转发到 funkeras，并发出 DeprecationWarning（计划 1.0.0 移除）。"""
    import sys

    # 允许重复 import 触发 warnings（模块可能已被其它测试预先加载过）。
    sys.modules.pop("notekeras", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        notekeras = importlib.import_module("notekeras")

    assert notekeras is funkeras
    deprecation_messages = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any("notekeras" in msg and "funkeras" in msg for msg in deprecation_messages)
    assert any("1.0.0" in msg for msg in deprecation_messages)
