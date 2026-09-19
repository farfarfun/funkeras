# Changelog

## [Unreleased]

### 新增

- 新增根目录 `tests/`，覆盖 `funkeras.exceptions`、`funkeras.models.rcnn.utils.nms` 等公开
  API 的正常路径与边界情况。
- 新增 `funkeras/exceptions.py`，定义 `FunkerasError`/`FeatureConfigError` 等领域异常类型。

### 修复

- `pyproject.toml` 补充运行时缺失的 `typeguard` 依赖（`funkeras/layers/wrappers.py` 一直依赖
  它，但此前从未声明，未手动安装 `typeguard` 时 `import funkeras` 会报
  `ModuleNotFoundError`），并为 `tensorflow`/`numpy`/`scipy`/`opencv-python`/`pillow`/`tqdm`/
  `typeguard`/`farlog` 全部补上版本下限，提交 `uv.lock`。
- `funkeras/features/feature_parse.py`、`feature_parse_keras.py` 中原先 `raise
  Exception("...")` 的几处配置校验，改为抛出 `FeatureConfigError` 并携带具体的配置片段作为
  上下文。
- `funkeras/models/rcnn/train.py`、`test.py`、`utils/anchor.py` 中的裸 `except:` 收窄为具体
  异常类型（`ValueError`/`IndexError`/`TypeError`），避免连 `KeyboardInterrupt`/`SystemExit`
  也被一并吞掉。
- `funkeras/models/rcnn/utils/nms.py`、`train.py` 中捕获 `Exception` 后仅 `print` 再继续/回退
  的几处，改为 `logger.exception` 保留堆栈，方便定位。
- 库代码里的诊断 `print()`/裸 `logging`（`funkeras/features/core.py`、
  `funkeras/models/similarity.py`、`funkeras/sample/gan/cyclegan/cyclegan.py`、
  `funkeras/layers/wrappers.py`、`funkeras/models/text.py`）统一改为 `farlog.getLogger`。

### 变更

- 移除 `script/build.sh`、`script/__version__.md`：旧脚本基于已不存在的 `setup.py`，且会在
  普通构建流程里无条件执行 `git pull`/`git add -A`/`git commit -a`/`git push`，有覆盖用户未
  提交改动、误推送的风险。仓库根目录已有 PEP 621 `pyproject.toml`，`funbuild` 会自动识别为
  `UVBuild` 并接管版本递增、构建、安装校验、发布、打标签的完整流程，不再需要手写脚本。
- `funkeras/models/text.py` 的公开函数 `textcnn` 补充 Python 3.10 风格类型标注
  （`model_img_path: str | None`）与中文 docstring，说明各参数含义与返回值。
- README 补充一句话简介、安装命令、最小可运行示例，并新增「第三方代码来源与协议」表格，
  逐项标注 `keras-self-attention`/`keras-transformer`/`keras-bert`/`keras-yolo3`/
  `keras-yolo3-detection`/`tensorflow/addons` 的原始协议与兼容性结论。
- `.gitignore` 补齐 `__pycache__/`、`*.db`、`*.rar`、`*.egg-info/`、`.venv/`、`.run/`、
  `logs/`、`.idea/`、`.vscode/`。

### 废弃

- `notekeras` 兼容层（`notekeras/__init__.py`）计划在 **1.0.0** 移除，请尽快把
  `import notekeras` / `from notekeras...` 换成 `import funkeras` / `from funkeras...`。
  `notekeras/__init__.py` 的 `DeprecationWarning` 文案已同步更新为写明目标版本。

## [0.9.0] - 2026-08-28

### 变更

- 包内导入路径从 `notekeras` 统一改为 `funkeras`，与仓库名、PyPI 发布名
  （一直都是 `funkeras`，当前已发布版本 0.8.12）保持一致。（破坏性变更）

### 废弃

- 保留了 `notekeras` 兼容层（`notekeras/__init__.py`，仅一个文件）：
  `import notekeras` 仍然可用，会转发到 `funkeras` 并抛出
  `DeprecationWarning`。计划在 1.0.0 中删除这个兼容层，请尽快把代码里的
  `import notekeras` / `from notekeras...` 换成
  `import funkeras` / `from funkeras...`。

### 已知问题（与本次改名无关的遗留问题，未处理）

- `funkeras/models/yolo4/core/utils.py`、`funkeras/models/yolo4/core/config.py`、
  `example/vgg/test.py`、`example/yolo/yolov3/*.py`、`example/yolo/yolov4/*.py` 中有若干
  硬编码的本机绝对路径（如 `/root/workspace/notechats/notekeras/...`、
  `/Users/liangtaoniu/workspace/MyDiary/notechats/notekeras/...`），指向作者本地开发机的
  目录结构，本来就无法在其他机器上直接运行，与本次导入名重命名无关，不在本次改动范围内。
