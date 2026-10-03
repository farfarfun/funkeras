# Changelog

## [Unreleased]

### 新增

- 新增根目录 `tests/`，覆盖 `funkeras.exceptions`、`funkeras.models.rcnn.utils.nms` 等公开
  API 的正常路径与边界情况。
- 新增 `funkeras/exceptions.py`，定义 `FunkerasError`/`FeatureConfigError` 等领域异常类型。
- 新增 `tests/test_ops_autodetect_mode.py`、`tests/test_parse_pandas_agg.py`、
  `tests/test_utils_file.py`，覆盖 `ops.autodetect_mode`、`features.parse_pandas` 的
  `agg_set`/`agg_list`、`utils.file.read_lines` 的正常路径与边界（不依赖 tensorflow 的用例
  直接运行，依赖 tensorflow 的用例用 `importorskip` 优雅跳过）。

### 修复

- `requires-python` 下限由 `>=3.10` 提升至 `>=3.11`：Python 3.10 环境下依赖解析只能拿到
  `keras==3.12.4`（受 GHSA 安全公告影响，< 3.15.0 均存在漏洞），3.11+ 才能解析到已修复的
  `keras==3.15.1`；提交更新后的 `uv.lock`。
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
- 继续把库代码里剩余的诊断 `print()` 改为 `farlog.getLogger`：
  `models/retinanet/generator.py`、`models/yolo3/core.py`、
  `sample/gan/wgan_gp/wgan_gp.py`、`temp/din/train.py`。
- `models/loader.py`、`activations/core.py`、`models/retinanet/losses.py` 的公开函数补齐
  Python 3.10 风格类型标注与中文 docstring。
- `utils/__init__.py` 改为按 PEP 562 惰性导入 `.image`（依赖 cv2）/`.util`（依赖
  tensorflow/matplotlib）子模块里的属性：此前 `__init__.py` 用 `from .image import *` /
  `from .util import *` eager 导入，导致仅仅 `from funkeras.utils import read_lines`
  这类零依赖调用也被迫要求安装 cv2/tensorflow，属于真实的导入期重依赖副作用 bug。
- `temp/din/train.py`：模块导入阶段原本会直接执行 `pickle.load`（读取本机数据集文件）和
  `train_model.fit_generator(...)`（发起训练），属于导入期文件 I/O/训练副作用的真实 bug；
  现已移入 `main()`，仅在 `python -m funkeras.temp.din.train` 直接运行时才执行；模块内
  `print` 改为 `farlog.getLogger`；并修正了从未生效过的错误导入路径
  `funkeras.din.*` → `funkeras.temp.din.*`。
- `example/yolo/yolov3/demo.py`：模块导入阶段原本会直接发起网络下载
  （`fundrive.lanzou.download`）、读取本机权重文件并运行推理，现移入 `setup()`/
  `if __name__ == "__main__":`，根目录路径改用环境变量 `FUNKERAS_YOLOV3_ROOT`。
- `example/yolo/yolov3/train.py`：修正导致脚本必定崩溃的遗留调试代码
  `a = b`（引用未定义变量，`NameError`）；根目录、权重路径、日志目录改用环境变量
  （`FUNKERAS_YOLOV3_ROOT`/`FUNKERAS_YOLOV3_WEIGHTS`/`FUNKERAS_YOLOV3_LOG_DIR`），不再
  硬编码作者本机路径；训练逻辑移入 `main()`，仅在直接运行时执行。
- `example/yolo/yolov4/detect.py`：模块导入阶段原本会直接发起网络下载，现移入
  `if __name__ == "__main__":`；根目录改用环境变量 `FUNKERAS_YOLOV4_ROOT`。
- 修正多个示例文件里指向已不存在的 `funkeras.model.*`（单数）模块路径的导入，改为实际的
  `funkeras.models.*`（复数），这些示例此前 import 阶段就会直接报错：
  `example/LoadExample.py`、`example/face/face.py`、`example/nlp/BertExample.py`、
  `example/nlp/TransExample.py`、`example/retinanet/examples/ResNet50RetinaNet.py`、
  `example/retinanet/examples/train.py`、`example/vgg/test.py`、
  `example/yolo/yolov3/demo.py`、`example/yolo/yolov3/train.py`、
  `example/yolo/yolov4/convert_tflite.py`、`example/yolo/yolov4/detect.py`、
  `example/yolo/yolov4/train.py`。
- `example/retinanet/examples/train.py`、`example/vgg/test.py`、
  `example/yolo/yolov4/train.py` 中硬编码的作者本机绝对路径（`/Users/liangtaoniu/...`、
  `/root/workspace/...`）改为环境变量（分别为
  `FUNKERAS_RETINANET_ANNOTATIONS`/`FUNKERAS_RETINANET_CLASSES`、`FUNKERAS_VGG_IMAGE`
  （默认回退到仓库自带示例图片）、`FUNKERAS_YOLO_ROOT`），未设置时给出清晰报错或可移植默认值。
- 移除已无代码引用、体积过大的生成物/第三方素材：`example/yolo/data/dataset/val2014.txt`
  （约 8.7 MB 的数据集清单）、`example/yolo/yolov3/results/yolo-{body,train}*.png`（推理可视
  化产物）、`src/funkeras/models/rcnn/results/*.png`（RCNN 推理结果图）、
  `src/funkeras/models/rcnn/font/ZiXinFangYunYuanTi-2.ttf`（约 8.7 MB 的中文字体，代码实际
  使用的是同目录下的 `FiraMono-Medium.otf`，该文件未被任何代码引用）；`.gitignore` 同步补充
  对应忽略规则，防止再次入库。
- GitHub 仓库 description 原文写着「导入名为 notekeras」，与实际情况相反（`notekeras` 只是
  已废弃的兼容转发层，主包名和实际导入名一直是 `funkeras`）；已更正为准确描述。

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
