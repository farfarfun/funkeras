# Changelog

## [0.9.0] - 2026-08-28

### Changed（破坏性变更）

- 包内导入路径从 `notekeras` 统一改为 `funkeras`，与仓库名、PyPI 发布名
  （一直都是 `funkeras`，当前已发布版本 0.8.12）保持一致。
- 保留了 `notekeras` 兼容层（`notekeras/__init__.py`，仅一个文件）：
  `import notekeras` 仍然可用，会转发到 `funkeras` 并抛出
  `DeprecationWarning`。计划在下一次破坏性版本中删除这个兼容层，请尽快把代码里的
  `import notekeras` / `from notekeras...` 换成
  `import funkeras` / `from funkeras...`。

### Known issues（已知的、与本次改名无关的遗留问题，未处理）

- `funkeras/models/yolo4/core/utils.py`、`funkeras/models/yolo4/core/config.py`、
  `example/vgg/test.py`、`example/yolo/yolov3/*.py`、`example/yolo/yolov4/*.py` 中有若干
  硬编码的本机绝对路径（如 `/root/workspace/notechats/notekeras/...`、
  `/Users/liangtaoniu/workspace/MyDiary/notechats/notekeras/...`），指向作者本地开发机的
  目录结构，本来就无法在其他机器上直接运行，与本次导入名重命名无关，不在本次改动范围内。
- `funkeras/layers/wrappers.py` 依赖 `typeguard`，但 `pyproject.toml` 一直没有声明
  这个依赖（改名前就是这样），导致 `import funkeras`（以及 `import notekeras` 兼容层）
  在没有额外手动安装 `typeguard` 的环境下会报 `ModuleNotFoundError`。这是改名前就存在的
  依赖声明缺口，本次未修复，仅在此记录。
