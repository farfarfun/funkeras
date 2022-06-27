from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.feature_column.feature_column_v2 import (
    DenseColumn, FeatureColumn, SequenceCategoricalColumn, SequenceDenseColumn)
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import sparse_ops
from tensorflow.python.util import deprecation

_FEATURE_COLUMN_DEPRECATION_DATE = None
_FEATURE_COLUMN_DEPRECATION = 'The old _FeatureColumn APIs are being deprecated. Please use the new FeatureColumn '


class IndicatorColumnDef(DenseColumn, SequenceDenseColumn, fc_old._DenseColumn, fc_old._SequenceDenseColumn):
    def __init__(self, categorical_column, size, *args):
        self.categorical_column = categorical_column
        self.size = size
        # super(IndicatorColumn2, self).__init__(*args)

    @property
    def _is_v2_column(self):
        return isinstance(self.categorical_column, FeatureColumn) and self.categorical_column._is_v2_column

    @property
    def name(self):
        """See `FeatureColumn` base class."""
        return '{}_indicator'.format(self.categorical_column.name)

    def _transform_id_weight_pair(self, id_weight_pair):
        id_tensor = id_weight_pair.id_tensor

        dense_id_tensor = sparse_ops.sparse_tensor_to_dense(
            id_tensor, default_value=-1)

        # One hot must be float for tf.concat reasons since all other inputs to input_layer are float32.
        # one_hot_id_tensor = array_ops.one_hot(dense_id_tensor, depth=self.categorical_column.num_buckets, on_value=1.0,
        #                                       off_value=0.0)

        # dense_id_tensor = math_ops.reduce_sum(dense_id_tensor, axis=[-2])
        return dense_id_tensor

        # return id_tensor

    def transform_feature(self, transformation_cache, state_manager):
        id_weight_pair = self.categorical_column.get_sparse_tensors(
            transformation_cache, state_manager)
        return self._transform_id_weight_pair(id_weight_pair)

    @property
    def parse_example_spec(self):
        """See `FeatureColumn` base class."""
        return self.categorical_column.parse_example_spec

    @property
    def variable_shape(self):
        """Returns a `TensorShape` representing the shape of the dense `Tensor`."""

        if isinstance(self.categorical_column, FeatureColumn):
            # return tensor_shape.TensorShape([1, self.categorical_column.num_buckets])
            # return tensor_shape.TensorShape([1, self.size])
            return tensor_shape.TensorShape(1)
        else:
            # return tensor_shape.TensorShape([1, self.categorical_column._num_buckets])
            return tensor_shape.TensorShape(1)

    def get_dense_tensor(self, transformation_cache, state_manager):
        if isinstance(self.categorical_column, SequenceCategoricalColumn):
            raise ValueError(
                'In indicator_column: {}. '
                'categorical_column must not be of type SequenceCategoricalColumn. '
                'Suggested fix A: If you wish to use DenseFeatures, use a '
                'non-sequence categorical_column_with_*. '
                'Suggested fix B: If you wish to create sequence input, use '
                'SequenceFeatures instead of DenseFeatures. '
                'Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        # Feature has been already transformed. Return the intermediate
        # representation created by transform_feature.
        return transformation_cache.get(self, state_manager)

    def get_sequence_dense_tensor(self, transformation_cache, state_manager):
        """See `SequenceDenseColumn` base class."""
        if not isinstance(self.categorical_column, SequenceCategoricalColumn):
            raise ValueError(
                'In indicator_column: {}. categorical_column must be of type SequenceCategoricalColumn '
                'to use SequenceFeatures. Suggested fix: Use one of sequence_categorical_column_with_*. '
                'Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
        # Feature has been already transformed. Return the intermediate representation created by transform_feature.
        dense_tensor = transformation_cache.get(self, state_manager)
        sparse_tensors = self.categorical_column.get_sparse_tensors(
            transformation_cache, state_manager)
        sequence_length = fc_utils.sequence_length_from_sparse_tensor(
            sparse_tensors.id_tensor)
        return SequenceDenseColumn.TensorSequenceLengthPair(dense_tensor=dense_tensor, sequence_length=sequence_length)

    @property
    def parents(self):
        """See 'FeatureColumn` base class."""
        return [self.categorical_column]

    def get_config(self):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import serialize_feature_column
        config = dict(zip(self._fields, self))
        config['categorical_column'] = serialize_feature_column(
            self.categorical_column)
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import \
            deserialize_feature_column  # pylint: disable=g-import-not-at-top
        _check_config_keys(config, cls._fields)
        kwargs = _standardize_and_copy_config(config)
        kwargs['categorical_column'] = deserialize_feature_column(config['categorical_column'], custom_objects,
                                                                  columns_by_name)
        return cls(**kwargs)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _transform_feature(self, inputs):
        id_weight_pair = self.categorical_column._get_sparse_tensors(
            inputs)  # pylint: disable=protected-access
        return self._transform_id_weight_pair(id_weight_pair)

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _variable_shape(self):
        return tensor_shape.TensorShape([1, self.categorical_column._num_buckets])  # pylint: disable=protected-access

    @property
    def parse_example_spec(self):
        """See `FeatureColumn` base class."""
        return self.categorical_column.parse_example_spec

    @property
    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _parse_example_spec(self):
        return self.categorical_column._parse_example_spec  # pylint: disable=protected-access

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        del weight_collections
        del trainable
        if isinstance(
                self.categorical_column,
                (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
            raise ValueError(
                'In indicator_column: {}. '
                'categorical_column must not be of type _SequenceCategoricalColumn. '
                'Suggested fix A: If you wish to use DenseFeatures, use a '
                'non-sequence categorical_column_with_*. '
                'Suggested fix B: If you wish to create sequence input, use '
                'SequenceFeatures instead of DenseFeatures. '
                'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                             self.categorical_column))
        # Feature has been already transformed. Return the intermediate
        # representation created by transform_feature.
        return inputs.get(self)

    @deprecation.deprecated(_FEATURE_COLUMN_DEPRECATION_DATE, _FEATURE_COLUMN_DEPRECATION)
    def _get_sequence_dense_tensor(self,
                                   inputs,
                                   weight_collections=None,
                                   trainable=None):
        # Do nothing with weight_collections and trainable since no variables are
        # created in this function.
        del weight_collections
        del trainable
        if not isinstance(
                self.categorical_column,
                (SequenceCategoricalColumn, fc_old._SequenceCategoricalColumn)):  # pylint: disable=protected-access
            raise ValueError(
                'In indicator_column: {}. '
                'categorical_column must be of type _SequenceCategoricalColumn '
                'to use SequenceFeatures. '
                'Suggested fix: Use one of sequence_categorical_column_with_*. '
                'Given (type {}): {}'.format(self.name, type(self.categorical_column),
                                             self.categorical_column))
        # Feature has been already transformed. Return the intermediate
        # representation created by _transform_feature.
        dense_tensor = inputs.get(self)
        sparse_tensors = self.categorical_column._get_sparse_tensors(
            inputs)  # pylint: disable=protected-access
        sequence_length = fc_utils.sequence_length_from_sparse_tensor(
            sparse_tensors.id_tensor)
        return SequenceDenseColumn.TensorSequenceLengthPair(
            dense_tensor=dense_tensor, sequence_length=sequence_length)


def _check_config_keys(config, expected_keys):
    """Checks that a config has all expected_keys."""
    if set(config.keys()) != set(expected_keys):
        raise ValueError(
            'Invalid config: {}, expected keys: {}'.format(config, expected_keys))


def _standardize_and_copy_config(config):
    """Returns a shallow copy of config with lists turned to tuples.

    Keras serialization uses nest to listify everything.
    This causes problems with the NumericColumn shape, which becomes
    unhashable. We could try to solve this on the Keras side, but that
    would require lots of tracking to avoid changing existing behavior.
    Instead, we ensure here that we revive correctly.

    Args:
      config: dict that will be used to revive a Feature Column

    Returns:
      Shallow copy of config with lists turned to tuples.
    """
    kwargs = config.copy()
    for k, v in kwargs.items():
        if isinstance(v, list):
            kwargs[k] = tuple(v)

    return kwargs
