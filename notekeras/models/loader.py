from __future__ import unicode_literals

import codecs
import json
import os
import shutil
from collections import namedtuple

import numpy as np
import tensorflow as tf

from notekeras.backend import keras
from notekeras.models import bert

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


def get_pre_trained_path(info=PreTrainedList.chinese_wwm_base):
    """
    获取模型路径
    :param info:模型路径
    :return:
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


def get_checkpoint_paths(model_path):
    config_path = os.path.join(model_path, 'bert_config.json')
    checkpoint_path = os.path.join(model_path, 'bert_model.ckpt')
    vocab_path = os.path.join(model_path, 'vocab.txt')
    return CheckpointPaths(config_path, checkpoint_path, vocab_path)


def get_checkpoint_config(info=PreTrainedList.chinese_wwm_base):
    """
    获取模型路径配置信息
    :param info:模型路径
    :return:
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


def build_model_from_config(config_file, training=False, trainable=None, output_layer_num=1, seq_len=int(1e9),
                            **kwargs):
    """Build the model from config file.

    :param config_file: The path to the JSON configuration file.
    :param training: If training, the whole model will be returned.
                     Otherwise, the MLM and NSP parts will be ignored.
    :param trainable: Whether the model is trainable.
    :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `training` is `False`.
    :param seq_len: If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    :return: model and config
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


def load_model_weights_from_checkpoint(model, config, checkpoint_file, training=False):
    """
    从checkpoint中加载官方的模型
    Load trained official model from checkpoint.

    :param model: Built keras model.
    :param config: 配置文件路径 Loaded configuration file.
    :param checkpoint_file:必须以.ckpt结尾的checkpoint文件路径 The path to the checkpoint files, should end with '.ckpt'.
    :param training: 如果需要训练，会返回整个模型
                    If training, the whole model will be returned.
                    Otherwise, the MLM and NSP parts will be ignored.
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
        except ValueError as e:
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


def load_trained_model_from_checkpoint(config_file, checkpoint_file, training=False, trainable=None,
                                       output_layer_num=1, seq_len=int(1e9), **kwargs):
    """Load trained official model from checkpoint.

    :param config_file: The path to the JSON configuration file.
    :param checkpoint_file: checkpoint_file.
    :param training: If training, the whole model will be returned.
                     Otherwise, the MLM and NSP parts will be ignored.
    :param trainable: Whether the model is trainable. The default value is the same with `training`.
    :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `training` is `False`.
    :param seq_len: If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    :return: model
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


def load_vocabulary(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


def checkpoint_loader(checkpoint_file):
    """
    从checkpoint加载变量
    :param checkpoint_file: 模型文件
    :return:
    """

    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)

    return _loader
