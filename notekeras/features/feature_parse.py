import pickle

import tensorflow as tf
from notekeras.layers import TrigPosEmbedding
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Concatenate, DenseFeatures, Embedding,
                                     Input, Layer)
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.feature_column.feature_column_lib import (
    NumericColumn, SequenceFeatures, numeric_column)

from .feature_column_def import IndicatorColumnDef

field_type_map = {
    'int': tf.dtypes.int32,
    'int32': tf.dtypes.int32,
    'int64': tf.dtypes.int64,
    'float': tf.dtypes.float32,
    'float32': tf.dtypes.float32,
    'float64': tf.dtypes.float64,
    'string': tf.dtypes.string,
}


def _parse_vocabulary(vocabulary):
    if isinstance(vocabulary, list):
        return vocabulary
    elif isinstance(vocabulary, str):
        return eval(vocabulary)
    else:
        return [i for i in vocabulary]


def _get_categorical_column(params: dict) -> fc.CategoricalColumn:
    if 'vocabulary' in params.keys():
        feature = fc.categorical_column_with_vocabulary_list(params['key'],
                                                             vocabulary_list=_parse_vocabulary(
                                                                 params['vocabulary']),
                                                             default_value=0)
    elif 'bucket_size' in params.keys():
        feature = fc.categorical_column_with_hash_bucket(params['key'],
                                                         hash_bucket_size=params['bucket_size'])
    elif 'file' in params.keys():
        feature = fc.categorical_column_with_vocabulary_file(params['key'],
                                                             vocabulary_file=params['file'],
                                                             default_value=0)
    elif 'num_buckets' in params.keys():
        feature = fc.categorical_column_with_identity(params['key'],
                                                      num_buckets=params['num_buckets'])
    elif 'boundaries' in params.keys():
        feature = fc.bucketized_column(fc.numeric_column(
            params['key']), boundaries=params['boundaries'])
    else:
        raise Exception("params error")

    return feature


def _get_sequence_categorical_column(params: dict) -> fc.SequenceCategoricalColumn:
    key = params['key']
    if 'vocabulary' in params.keys():
        feature = sfc.sequence_categorical_column_with_vocabulary_list(key,
                                                                       vocabulary_list=_parse_vocabulary(
                                                                           params['vocabulary']),
                                                                       default_value=0)
    elif 'bucket_size' in params.keys():
        feature = sfc.sequence_categorical_column_with_hash_bucket(
            key, hash_bucket_size=params['bucket_size'])
    elif 'file' in params.keys():
        feature = sfc.sequence_categorical_column_with_vocabulary_file(key,
                                                                       vocabulary_file=params['file'],
                                                                       default_value=0)
    elif 'num_buckets' in params.keys():
        feature = sfc.sequence_categorical_column_with_identity(key,
                                                                num_buckets=params['num_buckets'])
    else:
        raise Exception("params error")

    return feature


class ParseFeatureConfig:
    def __init__(self):
        self.feature_dict = {}
        self.share_layer = {}

    def _get_input_layer(self, params: dict, size=1) -> (str, Layer):
        field_name = params['key']
        if 'length' in params.keys():
            size = params['length']
        field_type = field_type_map.get(params['dtype'], tf.dtypes.string)

        if field_name in self.feature_dict.keys():
            inputs = self.feature_dict[field_name]
        else:
            inputs = Input((size,), dtype=field_type, name=field_name)
            self.feature_dict[field_name] = inputs

        return field_name, inputs

    def _get_share_layer(self, name: str, layer: Layer) -> Layer:
        """
        根据名称取出共享层
        :param name:
        :param layer:
        :return:
        """
        if name is None:
            return layer
        elif name in self.share_layer.keys():
            return self.share_layer[name]
        else:
            self.share_layer[name] = layer
            return layer

    def _numeric_column(self, params: dict) -> Layer:
        """
        输入：数值
        输出：数值
        :param params:
        :return:
        """
        key, inputs = self._get_input_layer(params)

        if 'transform' in params.keys():
            if params['transform'] == 'log':
                inputs = K.log(inputs)
            elif params['transform'] == 'sqrt':
                inputs = K.sqrt(inputs)

        #column = numeric_column(params['key'], shape=(params.get('length', 1),))
        #outputs = DenseFeatures(column)({key: inputs})
        return inputs

    def _cate_indicator_column(self, params: dict) -> DenseFeatures:
        """
        输入：类别
        输出：类别对应的one_hot
        :param params:
        :return:
        """
        key, inputs = self._get_input_layer(params)

        feature = _get_categorical_column(params)
        feature_column = fc.indicator_column(feature)

        outputs = DenseFeatures(
            feature_column, name=params.get('name', None))({key: inputs})

        return outputs

    def _cate_embedding_column(self, params: dict) -> Layer:
        """
        输入：类别
        输出：类别对应的embedding
        :param params:
        :return:
        """
        key, inputs = self._get_input_layer(params)

        feature = _get_categorical_column(params)

        column = IndicatorColumnDef(feature, size=1)

        sequence_input = DenseFeatures(column)({key: inputs})
        sequence_input = K.sum(sequence_input, axis=-1)

        name = params.get('share_name', None)
        layer = self._get_share_layer(name,
                                      Embedding(input_dim=feature.num_buckets + 1,
                                                output_dim=params['dimension'],
                                                mask_zero=True,
                                                # embeddings_regularizer=tf.keras.regularizers.l2(0.01),
                                                # activity_regularizer=tf.keras.regularizers.l2(0.01),
                                                name=name))
        res = layer(sequence_input)
        return res

    def _sequence_cate_indicator_column(self, params: dict):
        """
        输入：类别序列
        输出：类别序列对应的类别ID
        :param params:
        :return:
        """
        key, inputs = self._get_input_layer(params, size=params['length'])

        feature = _get_sequence_categorical_column(params)
        column = IndicatorColumnDef(feature, size=params['length'])

        sequence_input, sequence_length = SequenceFeatures(column)({
            key: inputs})

        return sequence_input, sequence_length

    def _sequence_cate_embedding_column(self, params: dict):
        """
        输入：类别序列
        输出：类别序列对应的embedding
        :param params:
        :return:
        """
        key, inputs = self._get_input_layer(params, size=params['length'])

        feature = _get_sequence_categorical_column(params)
        column = IndicatorColumnDef(feature, size=params['length'])

        sequence_input, sequence_length = SequenceFeatures(column)({
            key: inputs})

        sequence_input = tf.keras.backend.sum(sequence_input, axis=-1)

        name = params.get('share_name', None)
        layer = self._get_share_layer(name,
                                      Embedding(input_dim=feature.num_buckets + 1,
                                                output_dim=params['dimension'],
                                                mask_zero=True,
                                                weights=[pickle.load(open(params['weights'],
                                                                          'rb'))] if 'weights' in params.keys() else None,

                                                trainable=params.get(
                                                    'trainable', True),
                                                # embeddings_regularizer=tf.keras.regularizers.l2(0.01),
                                                # activity_regularizer=tf.keras.regularizers.l2(0.01),
                                                name=name))

        res = layer(sequence_input)
        res = TrigPosEmbedding(mode=TrigPosEmbedding.MODE_ADD, )(res)
        return res, sequence_length

    def _get_columns_map(self, key: str):
        _columns_map = {
            "NumericColumn": self._numeric_column,  # 数值类型
            "BucketizedColumn": self._cate_indicator_column,  # 分桶类型

            "CateIndicatorColumn": self._cate_indicator_column,  # 类别库生成的类别对应的one_hot
            "FileIndicatorColumn": self._cate_indicator_column,  # 读取文件产生的类别对应的one_hot
            "HashIndicatorColumn": self._cate_indicator_column,  # 关键词hash映射产生的类别对应的one_hot
            "BucketIndicatorColumn": self._cate_indicator_column,  # 数值分桶产生的类别对应的one_hot

            "CateEmbeddingColumn": self._cate_embedding_column,  # embedding
            "FileEmbeddingColumn": self._cate_embedding_column,
            "HashEmbeddingColumn": self._cate_embedding_column,
            "BucketEmbeddingColumn": self._cate_embedding_column,

            "SequenceCateIndicatorColumn": self._sequence_cate_indicator_column,
            "SequenceFileIndicatorColumn": self._sequence_cate_indicator_column,
            "SequenceHashIndicatorColumn": self._sequence_cate_indicator_column,

            "SequenceCateEmbddingColumn": self._sequence_cate_embedding_column,
            "SequenceFileEmbddingColumn": self._sequence_cate_embedding_column,
            "SequenceHashEmbddingColumn": self._sequence_cate_embedding_column
        }
        return _columns_map.get(key, None)

    def parse_feature_json(self, layer_json) -> Layer:
        outputs = []
        for feature_line in layer_json["inputs"]:
            feature_type_name = feature_line['type']
            feature_para = feature_line['parameters']

            method = self._get_columns_map(feature_type_name)
            if method is None:
                continue

            outputs.append(method(feature_para))

        #outputs = tf.keras.backend.concatenate(outputs)
        outputs = Concatenate()(outputs)

        return outputs

    def parse_sequence_feature_json(self, layer_json):
        feature_line = layer_json['inputs'][0]

        feature_type_name = feature_line['type']
        feature_para = feature_line['parameters']

        method = self._get_columns_map(feature_type_name)
        if method is None or not isinstance(feature_para, dict):
            raise Exception("error")

        sequence_input, sequence_length = method(feature_para)
        return sequence_input, sequence_length

    def parse_feature(self, layer_dict):
        values = []
        if isinstance(layer_dict, dict):
            values = layer_dict.values()
        elif isinstance(layer_dict, list):
            values = layer_dict

        outputs = []
        for value in values:
            if value['type'] == 'single':
                outputs.append(self.parse_feature_json(value))
            elif value['type'] == 'sequence':
                outputs.append(self.parse_sequence_feature_json(value))
        return outputs


def define_feature_json(key,
                        feature_type='NumericColumn',
                        feature_lenth=1,
                        dtype=None,
                        dimension=None,
                        share_name=None,
                        vocabulary=None,
                        bucket_size=None,
                        num_buckets=None,
                        *args, **kwargs):
    para = {
        "key": key,
        "length": feature_lenth,
        "dtype": dtype,
        "share_name": share_name,
        "dimension": dimension,
        "vocabulary": vocabulary,
        "bucket_size": bucket_size,
        "num_buckets": num_buckets
    }
    para.update(kwargs)
    para = {k: v for k, v in para.items() if k is not None and v is not None}

    return {
        "type": feature_type,
        "parameters": para
    }
