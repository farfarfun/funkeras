# funkeras

基于 TensorFlow/Keras 的模型、网络层与特征工程工具合集：包含 Attention/Transformer/BERT、
TextCNN、CycleGAN 等 GAN 系列、Faster R-CNN、YOLOv3 等模型的学习与整理实现，以及一批可复用的
自定义 Keras Layer（Attention、Embedding、FM 等）。

## 安装

```bash
pip install funkeras
```

## 最小示例

```python
import funkeras
from funkeras.layers import SeqSelfAttention
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# 用 funkeras 提供的 SeqSelfAttention 层搭一个最小的序列分类模型
inputs = Input(shape=(20, 32))
x = SeqSelfAttention(attention_activation="sigmoid")(inputs)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs, outputs)
model.summary()
```

`funkeras.models.text.textcnn` 提供了另一个开箱即用的公开 API 示例：

```python
from funkeras.models.text import textcnn

model = textcnn(
    max_sequence_length=60,
    max_token_num=5000,
    embedding_dim=300,
    output_dim=2,
)
```

## 内容说明

[my blog](http://blog.notechats.cn/)

`funkeras/layers/attention.py`、`funkeras/models/text.py` 等模块下的 attention、transformer、
bert 实现，主要是从大神 [CyberZHG](https://github.com/CyberZHG) 的
[keras-self-attention](https://github.com/CyberZHG/keras-self-attention)、
[keras-transformer](https://github.com/CyberZHG/keras-transformer)、
[keras-bert](https://github.com/CyberZHG/keras-bert) 学习来的，因为想研究源码并做一些标注，
所以进行了翻译、标注和整合。

`funkeras/models/yolo3/` 下的 YOLOv3 实现参考了
[keras-yolo3](https://github.com/qqwweee/keras-yolo3) 与
[keras-yolo3-detection](https://github.com/SpikeKing/keras-yolo3-detection)。

`funkeras/layers/wrappers.py` 里的 `WeightNormalization` 移植自
[tensorflow/addons](https://github.com/tensorflow/addons)。

### 第三方代码来源与协议

| 来源 | 协议 | 说明 |
| --- | --- | --- |
| [keras-self-attention](https://github.com/CyberZHG/keras-self-attention) | MIT | 与本项目 MIT 协议兼容 |
| [keras-transformer](https://github.com/CyberZHG/keras-transformer) | MIT | 与本项目 MIT 协议兼容 |
| [keras-bert](https://github.com/CyberZHG/keras-bert) | MIT | 与本项目 MIT 协议兼容 |
| [keras-yolo3](https://github.com/qqwweee/keras-yolo3)（`qqwweee`） | MIT | 与本项目 MIT 协议兼容 |
| [keras-yolo3-detection](https://github.com/SpikeKing/keras-yolo3-detection)（`SpikeKing`） | 未标注协议 | 上游仓库未附 LICENSE 文件，无法确认协议兼容性；若计划在自己的项目中使用 `funkeras/models/yolo3/` 下依赖该来源的部分，请先联系原作者确认授权 |
| [tensorflow/addons](https://github.com/tensorflow/addons)（`WeightNormalization`） | Apache-2.0 | 与本项目 MIT 协议兼容（Apache-2.0 允许被 MIT 项目吸收，需保留版权声明） |

# Attention

## ScaledDotProductAttention
[参考论文](https://arxiv.org/pdf/1706.03762.pdf)

$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V$$

## SeqSelfAttention
[参考论文](https://arxiv.org/pdf/1806.01264.pdf)

### multiplicative
$$e_{t, t'} = x_t^T W_a x_{t'} + b_a$$

$$a_{t} = \text{softmax}(e_t)$$

$$l_t = \sum_{t'} a_{t, t'} x_{t'}$$

### additive
$$h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)$$

$$e_{t, t'} = W_a h_{t, t'} + b_a$$

$$a_{t} = \text{softmax}(e_t)$$

$$l_t = \sum_{t'} a_{t, t'} x_{t'}$$


## SeqWeightedAttention
[参考论文](https://arxiv.org/pdf/1708.00524.pdf)

$$Y = \text{softmax}(XW + b) X$$


## MultiHeadAttention
[参考论文](https://arxiv.org/pdf/1706.03762.pdf)

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
