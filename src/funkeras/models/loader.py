from __future__ import unicode_literals

import codecs
import json
import os
import shutil
from collections import namedtuple
from collections.abc import Callable
from typing import Any

import numpy as np
import tensorflow as tf

from funkeras.backend import keras
from funkeras.models import bert

__all__ = [
    'build_model_from_config',
    'load_model_weights_from_checkpoint',
    'load_trained_model_from_checkpoint',
    'load_vocabulary',
    'PreTrainedInfo',
    'PreTrainedList',
    'get_pre_trained_path',
    'get_checkpoint_paths',
    'get_checkpoint_config',
]

PreTrainedInfo = namedtuple('PreTrainedInfo', ['url', 'extract_name', 'target_name'])
CheckpointPaths = namedtuple('CheckpointPaths', ['config', 'checkpoint', 'vocab'])


class PreTrainedList(object):
    __test__ = PreTrainedInfo(
        'https://github.com/CyberZHG/keras-bert/archive/master.zip',
        'keras-bert-master',
        'keras-bert',
    )

    multi_cased_base = 'https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip'
    chinese_base = 'https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip'
    wwm_uncased_large = 'https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip'
    wwm_cased_large = 'https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip'
    chinese_wwm_base = PreTrainedInfo(
        'https://storage.googleapis.com/hfl-rc/chinese-bert/chinese_wwm_L-12_H-768_A-12.zip',
        'publish',
        'chinese_wwm_L-12_H-768_A-12',
    )


def get_pre_trained_path(
    info: PreTrainedInfo | str = PreTrainedList.chinese_wwm_base,
) -> str:
    """下载（如未缓存）并返回预训练模型的本地解压目录。

    参数:
        info: 预训练模型描述，可以是 ``PreTrainedInfo`` 具名元组，也可以直接是
            模型压缩包的下载地址字符串。

    返回:
        预训练模型解压后所在的本地目录路径。
    """
    path = info
    if isinstance(info, PreTrainedInfo):
        path = info.url
    path = keras.utils.get_file(fname=os.path.split(path)[-1], origin=path, extract=True)
    base_part, file_part = os.path.split(path)
    file_part = file_part.split('.')[0]
    if isinstance(info, PreTrainedInfo):
        extract_path = os.path.join(base_part, info.extract_name)
        target_path = os.path.join(base_part, info.target_name)
        if not os.path.exists(target_path):
            shutil.move(extract_path, target_path)
        file_part = info.target_name
    return os.path.join(base_part, file_part)


def get_checkpoint_paths(model_path: str) -> CheckpointPaths:
    """根据模型目录拼出 BERT checkpoint 的标准三件套路径。

    参数:
        model_path: 预训练模型所在目录。

    返回:
        包含 ``config``/``checkpoint``/``vocab`` 三个路径的具名元组。
    """
    config_path = os.path.join(model_path, 'bert_config.json')
    checkpoint_path = os.path.join(model_path, 'bert_model.ckpt')
    vocab_path = os.path.join(model_path, 'vocab.txt')
    return CheckpointPaths(config_path, checkpoint_path, vocab_path)


def get_checkpoint_config(
    info: PreTrainedInfo | str = PreTrainedList.chinese_wwm_base,
) -> CheckpointPaths:
    """下载（如未缓存）预训练模型并返回其 checkpoint 路径配置。

    参数:
        info: 预训练模型描述，语义同 :func:`get_pre_trained_path`。

    返回:
        包含 ``config``/``checkpoint``/``vocab`` 三个路径的具名元组。
    """
    path = info
    if isinstance(info, PreTrainedInfo):
        path = info.url
    path = keras.utils.get_file(fname=os.path.split(path)[-1], origin=path, extract=True)
    base_part, file_part = os.path.split(path)
    file_part = file_part.split('.')[0]
    if isinstance(info, PreTrainedInfo):
        extract_path = os.path.join(base_part, info.extract_name)
        target_path = os.path.join(base_part, info.target_name)
        if not os.path.exists(target_path):
            shutil.move(extract_path, target_path)
        file_part = info.target_name

    model_path = os.path.join(base_part, file_part)
    config_path = os.path.join(model_path, 'bert_config.json')
    checkpoint_path = os.path.join(model_path, 'bert_model.ckpt')
    vocab_path = os.path.join(model_path, 'vocab.txt')
    return CheckpointPaths(config_path, checkpoint_path, vocab_path)


def build_model_from_config(
    config_file: str,
    training: bool = False,
    trainable: bool | None = None,
    output_layer_num: int = 1,
    seq_len: int | None = int(1e9),
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """根据 BERT 配置文件构建模型。

    参数:
        config_file: JSON 配置文件路径。
        training: 是否用于训练；为 ``True`` 时返回完整模型（含 MLM/NSP 头），
            否则只返回主干模型。
        trainable: 模型权重是否可训练，默认与 ``training`` 一致。
        output_layer_num: 取最后多少层输出做拼接，仅在 ``training=False`` 时生效。
        seq_len: 若小于配置文件中的位置编码长度，会据此截断位置编码权重；传
            ``None`` 表示不做截断。
        **kwargs: 透传给 :func:`funkeras.models.bert.get_model` 的额外参数。

    返回:
        ``(model, config)`` 二元组：构建好的 Keras 模型与解析后的配置字典。
    """
    with open(config_file, 'r') as reader:
        config = json.loads(reader.read())
    if seq_len is not None:
        config['max_position_embeddings'] = seq_len = min(seq_len, config['max_position_embeddings'])
    if trainable is None:
        trainable = training
    model = bert.get_model(token_num=config['vocab_size'],
                           pos_num=config['max_position_embeddings'],
                           seq_len=seq_len,
                           embed_dim=config['hidden_size'],
                           transformer_num=config['num_hidden_layers'],
                           head_num=config['num_attention_heads'],
                           feed_forward_dim=config['intermediate_size'],
                           feed_forward_activation=config['hidden_act'],
                           training=training,
                           trainable=trainable,
                           output_layer_num=output_layer_num,
                           **kwargs)
    if not training:
        inputs, outputs = model
        model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model, config


def load_model_weights_from_checkpoint(
    model: Any,
    config: dict[str, Any],
    checkpoint_file: str,
    training: bool = False,
) -> None:
    """从官方 checkpoint 文件中加载权重到已构建好的模型。

    参数:
        model: 已通过 :func:`build_model_from_config` 构建好的 Keras 模型。
        config: 对应的配置字典（需包含 ``num_hidden_layers`` 等字段）。
        checkpoint_file: checkpoint 文件路径，须以 ``.ckpt`` 结尾。
        training: 为 ``True`` 时额外加载 MLM/NSP 头的权重，否则只加载主干权重。
    """
    loader = checkpoint_loader(checkpoint_file)

    model.get_layer(name='Embedding-Token').set_weights([
        loader('bert/embeddings/word_embeddings'),
    ])
    model.get_layer(name='Embedding-Position').set_weights([
        loader('bert/embeddings/position_embeddings')[:config['max_position_embeddings'], :],
    ])
    model.get_layer(name='Embedding-Segment').set_weights([
        loader('bert/embeddings/token_type_embeddings'),
    ])
    model.get_layer(name='Embedding-Norm').set_weights([
        loader('bert/embeddings/LayerNorm/gamma'),
        loader('bert/embeddings/LayerNorm/beta'),
    ])
    for i in range(config['num_hidden_layers']):
        try:
            model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1))
        except ValueError:
            continue
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/attention/self/query/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/query/bias' % i),
            loader('bert/encoder/layer_%d/attention/self/key/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/key/bias' % i),
            loader('bert/encoder/layer_%d/attention/self/value/kernel' % i),
            loader('bert/encoder/layer_%d/attention/self/value/bias' % i),
            loader('bert/encoder/layer_%d/attention/output/dense/kernel' % i),
            loader('bert/encoder/layer_%d/attention/output/dense/bias' % i),
        ])
        model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
            loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
        ])
        model.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/intermediate/dense/kernel' % i),
            loader('bert/encoder/layer_%d/intermediate/dense/bias' % i),
            loader('bert/encoder/layer_%d/output/dense/kernel' % i),
            loader('bert/encoder/layer_%d/output/dense/bias' % i),
        ])
        model.get_layer(name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
            loader('bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
            loader('bert/encoder/layer_%d/output/LayerNorm/beta' % i),
        ])
    if training:
        model.get_layer(name='MLM-Dense').set_weights([
            loader('cls/predictions/transform/dense/kernel'),
            loader('cls/predictions/transform/dense/bias'),
        ])
        model.get_layer(name='MLM-Norm').set_weights([
            loader('cls/predictions/transform/LayerNorm/gamma'),
            loader('cls/predictions/transform/LayerNorm/beta'),
        ])
        model.get_layer(name='MLM-Sim').set_weights([
            loader('cls/predictions/output_bias'),
        ])
        model.get_layer(name='NSP-Dense').set_weights([
            loader('bert/pooler/dense/kernel'),
            loader('bert/pooler/dense/bias'),
        ])
        model.get_layer(name='NSP').set_weights([
            np.transpose(loader('cls/seq_relationship/output_weights')),
            loader('cls/seq_relationship/output_bias'),
        ])


def load_trained_model_from_checkpoint(
    config_file: str,
    checkpoint_file: str,
    training: bool = False,
    trainable: bool | None = None,
    output_layer_num: int = 1,
    seq_len: int | None = int(1e9),
    **kwargs: Any,
) -> Any:
    """根据配置文件构建模型并从 checkpoint 加载权重，一步到位。

    参数:
        config_file: JSON 配置文件路径。
        checkpoint_file: checkpoint 文件路径，须以 ``.ckpt`` 结尾。
        training: 是否用于训练；为 ``True`` 时返回完整模型。
        trainable: 模型权重是否可训练，默认与 ``training`` 一致。
        output_layer_num: 取最后多少层输出做拼接，仅在 ``training=False`` 时生效。
        seq_len: 若小于配置文件中的位置编码长度，会据此截断位置编码权重。
        **kwargs: 透传给 :func:`build_model_from_config` 的额外参数。

    返回:
        加载好权重的 Keras 模型。
    """
    # model_path = get_pre_trained_path(model_info)
    # paths = get_checkpoint_paths(model_path)
    # config_file = paths.config
    # checkpoint_file = paths.checkpoint

    # 创建模型
    model, config = build_model_from_config(config_file, training=training, trainable=trainable,
                                            output_layer_num=output_layer_num, seq_len=seq_len, **kwargs)

    # 从checkpoint中加载网络权重
    load_model_weights_from_checkpoint(model, config, checkpoint_file, training=training)
    return model


def load_vocabulary(vocab_path: str) -> dict[str, int]:
    """加载 BERT 词表文件，构建 token 到 id 的映射。

    参数:
        vocab_path: 词表文件路径，每行一个 token。

    返回:
        token 到自增 id 的映射字典（id 即 token 在文件中的行号，从 0 开始）。
    """
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


def checkpoint_loader(checkpoint_file: str) -> Callable[[str], Any]:
    """返回一个按变量名从 checkpoint 读取权重的加载函数。

    参数:
        checkpoint_file: checkpoint 文件路径。

    返回:
        接收变量名、返回对应权重数组的可调用对象。
    """

    def _loader(name: str) -> Any:
        return tf.train.load_variable(checkpoint_file, name)

    return _loader
