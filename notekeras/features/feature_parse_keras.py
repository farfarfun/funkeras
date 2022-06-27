"""
参考网址
https://github.com/tensorflow/community/blob/master/rfcs/20191212-keras-categorical-inputs.md
"""
import tensorflow as tf
from notekeras.layers.core import SelfSum
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate, Embedding, Input, Layer
from tensorflow.keras.layers.experimental.preprocessing import (
    CategoryEncoding, Hashing, IntegerLookup, StringLookup)

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


class ParseFeatureConfig:
    embeddings_regularizer = 1e-4
    activity_regularizer = 1e-4

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

        return inputs

    def _category_hash(self, params: dict):
        """
        Replacing tf.feature_column.categorical_column_with_hash_bucket with Hashing from
        :param params:
        :return:
        """
        key, input_layer = self._get_input_layer(params)
        return Hashing(num_bins=params['hash_bucket_size'])(input_layer)

    def _category_lookup(self, params: dict):
        key, input_layer = self._get_input_layer(params)
        num_oov_buckets = params.get('num_oov_buckets', 0)
        if input_layer.dtype == 'string':
            if 'vocabulary_file' in params.keys():
                return StringLookup(max_tokens=params['vocabulary_size'],
                                    num_oov_indices=num_oov_buckets,
                                    mask_token=None,
                                    vocabulary=params['vocabulary_file'])(input_layer)
            elif 'vocabulary_list' in params.keys():
                return StringLookup(max_tokens=len(params['vocabulary_list']) + num_oov_buckets,
                                    num_oov_indices=num_oov_buckets,
                                    mask_token=None,
                                    vocabulary=params['vocabulary_list'])(input_layer)
        else:
            if 'vocabulary_file' in params.keys():
                return IntegerLookup(max_values=params['vocabulary_size'] + num_oov_buckets,
                                     num_oov_indices=num_oov_buckets,
                                     mask_value=None,
                                     vocabulary=['vocabulary_file'])(input_layer)
            elif 'vocabulary_list' in params.keys():
                return IntegerLookup(max_values=len(params['vocabulary_list']) + num_oov_buckets,
                                     num_oov_indices=num_oov_buckets,
                                     mask_value=None,
                                     vocabulary=params['vocabulary_list'])(input_layer)

    def _category_onehot(self, params: dict):
        if params['dtype'] in ('int', 'int32', 'int64'):
            num_buckets = params['num_buckets']
            key, input_layer = self._get_input_layer(params)
        else:
            input_layer = self._category_lookup(params)
            num_buckets = len(params['vocabulary_list'])

        name = params.get('name', params['key'] + '-onehot')
        cate_encode = CategoryEncoding(
            max_tokens=num_buckets, output_mode="binary", name=name)
        output = cate_encode(input_layer)
        return output

    def _category_embedding(self, params: dict):
        if params['dtype'] in ('int', 'int32', 'int64'):
            vocabulary_size = params['vocabulary_size']
            key, input_layer = self._get_input_layer(params)
        else:
            input_layer = self._category_lookup(params)
            vocabulary_size = len(params['vocabulary_list'])

        name = params.get('share_name', params['key'] + '-emb')
        embedding_layer = self._get_share_layer(name, Embedding(name=name,
                                                                input_dim=vocabulary_size,
                                                                output_dim=params['dimension'],
                                                                trainable=True,
                                                                mask_zero=True,
                                                                embeddings_regularizer=tf.keras.regularizers.l2(
                                                                    self.embeddings_regularizer),
                                                                activity_regularizer=tf.keras.regularizers.l2(
                                                                    self.activity_regularizer),
                                                                ))
        embedding_input = embedding_layer(input_layer)

        embedding_input = SelfSum(
            axis=-2, name=params['key'] + 'sum')(embedding_input)

        return embedding_input

    def _category_indicate(self, params: dict, weight_input: Layer = None):
        """
        Replacing tf.feature_column.indicator_column with CategoryEncoding from
        :param params:
        :param weight_input:
        :return:
        """
        id_input = self._category_lookup(params)
        if weight_input is None:
            encoded_input = CategoryEncoding(max_tokens=params['num_buckets'], output_mode="count", sparse=True)(
                id_input)
        else:
            encoded_input = CategoryEncoding(max_tokens=params['num_buckets'], output_mode="count", sparse=True)(
                id_input, weight_input)
        return encoded_input

    def _sequence_cate_indicator_column(self):
        pass

    def _sequence_cate_embedding_column(self):
        pass

    def _get_columns_map(self, key: str):
        _columns_map = {
            "NumericColumn": self._numeric_column,  # 数值类型
            "BucketColumn": self._category_hash,  # 分桶类型

            "CateIndicatorColumn": self._category_indicate,  # 类别库生成的类别对应的one_hot
            "FileIndicatorColumn": self._category_indicate,  # 读取文件产生的类别对应的one_hot
            "HashIndicatorColumn": self._category_indicate,  # 关键词hash映射产生的类别对应的one_hot
            "BucketIndicatorColumn": self._category_indicate,  # 数值分桶产生的类别对应的one_hot

            "CateOneHotColumn": self._category_onehot,  # one-hot

            "CateEmbeddingColumn": self._category_embedding,  # embedding
            "FileEmbeddingColumn": self._category_embedding,
            "HashEmbeddingColumn": self._category_embedding,
            "BucketEmbeddingColumn": self._category_embedding,

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
        if len(outputs) > 1:
            outputs = Concatenate()(outputs)
        elif len(outputs) == 1:
            outputs = outputs[0]
        else:
            raise Exception("Empty")

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
