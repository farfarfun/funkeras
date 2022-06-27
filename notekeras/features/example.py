from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.python.client import session
from tensorflow.python.feature_column import dense_features
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.feature_column import serialization
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops, errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops, sparse_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test

test_case = test.TestCase()


def _assert_sparse_tensor_indices_shape(expected, actual):
    test_case.assertEqual(np.int64, np.array(actual.indices).dtype)
    test_case.assertAllEqual(expected.indices, actual.indices)

    test_case.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
    test_case.assertAllEqual(expected.dense_shape, actual.dense_shape)


def _assert_sparse_tensor_value(expected, actual):
    _assert_sparse_tensor_indices_shape(expected, actual)

    test_case.assertEqual(np.array(expected.values).dtype, np.array(actual.values).dtype)
    test_case.assertAllEqual(expected.values, actual.values)


def _get_sequence_dense_tensor(column, features):
    return column.get_sequence_dense_tensor(fc.FeatureTransformationCache(features), None)


def _get_sequence_dense_tensor_state(column, features):
    state_manager = fc._StateManagerImpl(Layer(), trainable=True)
    column.create_state(state_manager)
    dense_tensor, lengths = column.get_sequence_dense_tensor(fc.FeatureTransformationCache(features), state_manager)
    return dense_tensor, lengths, state_manager


def _get_sparse_tensors(column, features):
    return column.get_sparse_tensors(
        fc.FeatureTransformationCache(features), None)


def _initialized_session(config=None):
    sess = session.Session(config=config)
    sess.run(variables_lib.global_variables_initializer())
    sess.run(lookup_ops.tables_initializer())
    return sess


class SequenceFeaturesTes:
    def run(self):
        params = [
            {'testcase_name': '2D',
             'sparse_input_args_a': {
                 # example 0, ids [2]
                 # example 1, ids [0, 1]
                 'indices': ((0, 0), (1, 0), (1, 1)),
                 'values': (2, 0, 1),
                 'dense_shape': (2, 2)},
             'sparse_input_args_b': {
                 # example 0, ids [1]
                 # example 1, ids [2, 0]
                 'indices': ((0, 0), (1, 0), (1, 1)),
                 'values': (1, 2, 0),
                 'dense_shape': (2, 2)},
             'expected_input_layer': [
                 # example 0, ids_a [2], ids_b [1]
                 [[5., 6., 14., 15., 16.], [0., 0., 0., 0., 0.]],
                 # example 1, ids_a [0, 1], ids_b [2, 0]
                 [[1., 2., 17., 18., 19.], [3., 4., 11., 12., 13.]], ],
             'expected_sequence_length': [1, 2]},
            {'testcase_name': '3D',
             'sparse_input_args_a': {
                 # feature 0, ids [[2], [0, 1]]
                 # feature 1, ids [[0, 0], [1]]
                 'indices': (
                     (0, 0, 0), (0, 1, 0), (0, 1, 1),
                     (1, 0, 0), (1, 0, 1), (1, 1, 0)),
                 'values': (2, 0, 1, 0, 0, 1),
                 'dense_shape': (2, 2, 2)},
             'sparse_input_args_b': {
                 # feature 0, ids [[1, 1], [1]]
                 # feature 1, ids [[2], [0]]
                 'indices': ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
                 'values': (1, 1, 1, 2, 0),
                 'dense_shape': (2, 2, 2)},
             'expected_input_layer': [
                 # feature 0, [a: 2, -, b: 1, 1], [a: 0, 1, b: 1, -]
                 [[5., 6., 14., 15., 16.], [2., 3., 14., 15., 16.]],
                 # feature 1, [a: 0, 0, b: 2, -], [a: 1, -, b: 0, -]
                 [[1., 2., 17., 18., 19.], [3., 4., 11., 12., 13.]]],
             'expected_sequence_length': [2, 2]},
        ]
        for param in params:
            self._embedding_column(
                sparse_input_args_a=param['sparse_input_args_a'],
                sparse_input_args_b=param['sparse_input_args_b'],
                expected_input_layer=param['expected_input_layer'],
                expected_sequence_length=param['expected_sequence_length'],
            )

        params = [
            {'testcase_name': '2D',
             'sparse_input_args_a': {
                 # example 0, ids [2]
                 # example 1, ids [0, 1]
                 'indices': ((0, 0), (1, 0), (1, 1)),
                 'values': (2, 0, 1),
                 'dense_shape': (2, 2)},
             'sparse_input_args_b': {
                 # example 0, ids [1]
                 # example 1, ids [1, 0]
                 'indices': ((0, 0), (1, 0), (1, 1)),
                 'values': (1, 1, 0),
                 'dense_shape': (2, 2)},
             'expected_input_layer': [
                 # example 0, ids_a [2], ids_b [1]
                 [[0., 0., 1., 0., 1.], [0., 0., 0., 0., 0.]],
                 # example 1, ids_a [0, 1], ids_b [1, 0]
                 [[1., 0., 0., 0., 1.], [0., 1., 0., 1., 0.]]],
             'expected_sequence_length': [1, 2]},
            {'testcase_name': '3D',
             'sparse_input_args_a': {
                 # feature 0, ids [[2], [0, 1]]
                 # feature 1, ids [[0, 0], [1]]
                 'indices': (
                     (0, 0, 0), (0, 1, 0), (0, 1, 1),
                     (1, 0, 0), (1, 0, 1), (1, 1, 0)),
                 'values': (2, 0, 1, 0, 0, 1),
                 'dense_shape': (2, 2, 2)},
             'sparse_input_args_b': {
                 # feature 0, ids [[1, 1], [1]]
                 # feature 1, ids [[1], [0]]
                 'indices': ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
                 'values': (1, 1, 1, 1, 0),
                 'dense_shape': (2, 2, 2)},
             'expected_input_layer': [
                 # feature 0, [a: 2, -, b: 1, 1], [a: 0, 1, b: 1, -]
                 [[0., 0., 1., 0., 2.], [1., 1., 0., 0., 1.]],
                 # feature 1, [a: 0, 0, b: 1, -], [a: 1, -, b: 0, -]
                 [[2., 0., 0., 0., 1.], [0., 1., 0., 1., 0.]]],
             'expected_sequence_length': [2, 2]},
        ]
        for param in params:
            self._indicator_column(
                sparse_input_args_a=param['sparse_input_args_a'],
                sparse_input_args_b=param['sparse_input_args_b'],
                expected_input_layer=param['expected_input_layer'],
                expected_sequence_length=param['expected_sequence_length'],
            )

        params = [
            {'testcase_name': '2D',
             'sparse_input_args': {
                 # example 0, values [0., 1]
                 # example 1, [10.]
                 'indices': ((0, 0), (0, 1), (1, 0)),
                 'values': (0., 1., 10.),
                 'dense_shape': (2, 2)},
             'expected_input_layer': [
                 [[0.], [1.]],
                 [[10.], [0.]]],
             'expected_sequence_length': [2, 1]},
            {'testcase_name': '3D',
             'sparse_input_args': {
                 # feature 0, ids [[20, 3], [5]]
                 # feature 1, ids [[3], [8]]
                 'indices': ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
                 'values': (20., 3., 5., 3., 8.),
                 'dense_shape': (2, 2, 2)},
             'expected_input_layer': [
                 [[20.], [3.], [5.], [0.]],
                 [[3.], [0.], [8.], [0.]]],
             'expected_sequence_length': [2, 2]},
        ]
        for param in params:
            self._numeric_column_multi_dim(
                sparse_input_args=param['sparse_input_args'],
                expected_input_layer=param['expected_input_layer'],
                expected_sequence_length=param['expected_sequence_length'],
            )

        params = [
            {'testcase_name': '2D',
             'sparse_input_args': {
                 # example 0, values [0., 1.,  2., 3., 4., 5., 6., 7.]
                 # example 1, [10., 11., 12., 13.]
                 'indices': ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                             (0, 7), (1, 0), (1, 1), (1, 2), (1, 3)),
                 'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
                 'dense_shape': (2, 8)},
             'expected_input_layer': [
                 # The output of numeric_column._get_dense_tensor should be flattened.
                 [[0., 1., 2., 3.], [4., 5., 6., 7.]],
                 [[10., 11., 12., 13.], [0., 0., 0., 0.]]],
             'expected_sequence_length': [2, 1]},
            {'testcase_name': '3D',
             'sparse_input_args': {
                 # example 0, values [[0., 1., 2., 3.]], [[4., 5., 6., 7.]]
                 # example 1, [[10., 11., 12., 13.], []]
                 'indices': ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3),
                             (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
                             (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3)),
                 'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
                 'dense_shape': (2, 2, 4)},
             'expected_input_layer': [
                 # The output of numeric_column._get_dense_tensor should be flattened.
                 [[0., 1., 2., 3.], [4., 5., 6., 7.]],
                 [[10., 11., 12., 13.], [0., 0., 0., 0.]]],
             'expected_sequence_length': [2, 1]},
        ]
        for param in params:
            self._numeric_column(
                sparse_input_args=param['sparse_input_args'],
                expected_input_layer=param['expected_input_layer'],
                expected_sequence_length=param['expected_sequence_length'],
            )

        params = [
            {'testcase_name': '2D',
             'sparse_input_args': {
                 # example 0, values [[[0., 1.],  [2., 3.]], [[4., 5.],  [6., 7.]]]
                 # example 1, [[[10., 11.],  [12., 13.]]]
                 'indices': ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                             (0, 7), (1, 0), (1, 1), (1, 2), (1, 3)),
                 'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
                 'dense_shape': (2, 8)},
             'expected_shape': [2, 2, 4]},
            {'testcase_name': '3D',
             'sparse_input_args': {
                 # example 0, values [[0., 1., 2., 3.]], [[4., 5., 6., 7.]]
                 # example 1, [[10., 11., 12., 13.], []]
                 'indices': ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3),
                             (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3),
                             (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3)),
                 'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
                 'dense_shape': (2, 2, 4)},
             'expected_shape': [2, 2, 4]},
        ]
        for param in params:
            self._static_shape_from_tensors_numeric(
                sparse_input_args=param['sparse_input_args'],
                expected_shape=param['expected_shape'],
            )

        params = [
            {'testcase_name': '2D',
             'sparse_input_args': {
                 # example 0, ids [2]
                 # example 1, ids [0, 1]
                 # example 2, ids []
                 # example 3, ids [1]
                 'indices': ((0, 0), (1, 0), (1, 1), (3, 0)),
                 'values': (2, 0, 1, 1),
                 'dense_shape': (4, 2)},
             'expected_shape': [4, 2, 3]},
            {'testcase_name': '3D',
             'sparse_input_args': {
                 # example 0, ids [[2]]
                 # example 1, ids [[0, 1], [2]]
                 # example 2, ids []
                 # example 3, ids [[1], [0, 2]]
                 'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0),
                             (3, 0, 0), (3, 1, 0), (3, 1, 1)),
                 'values': (2, 0, 1, 2, 1, 0, 2),
                 'dense_shape': (4, 2, 2)},
             'expected_shape': [4, 2, 3]}
        ]
        for param in params:
            self._static_shape_from_tensors_indicator(
                sparse_input_args=param['sparse_input_args'],
                expected_shape=param['expected_shape'],
            )

    def _embedding_column(self, sparse_input_args_a, sparse_input_args_b, expected_input_layer,
                          expected_sequence_length):
        sparse_input_a = sparse_tensor.SparseTensorValue(**sparse_input_args_a)
        sparse_input_b = sparse_tensor.SparseTensorValue(**sparse_input_args_b)
        vocabulary_size = 3
        embedding_dimension_a = 2
        embedding_values_a = (
            (1., 2.),  # id 0
            (3., 4.),  # id 1
            (5., 6.)  # id 2
        )
        embedding_dimension_b = 3
        embedding_values_b = (
            (11., 12., 13.),  # id 0
            (14., 15., 16.),  # id 1
            (17., 18., 19.)  # id 2
        )

        def _get_initializer(embedding_dimension, embedding_values):
            def _initializer(shape, dtype, partition_info=None):
                test_case.assertAllEqual((vocabulary_size, embedding_dimension), shape)
                test_case.assertEqual(dtypes.float32, dtype)
                test_case.assertIsNone(partition_info)
                return embedding_values

            return _initializer

        categorical_column_a = sfc.sequence_categorical_column_with_identity(
            key='aaa', num_buckets=vocabulary_size)
        embedding_column_a = fc.embedding_column(
            categorical_column_a,
            dimension=embedding_dimension_a,
            initializer=_get_initializer(embedding_dimension_a, embedding_values_a))
        categorical_column_b = sfc.sequence_categorical_column_with_identity(
            key='bbb', num_buckets=vocabulary_size)
        embedding_column_b = fc.embedding_column(
            categorical_column_b,
            dimension=embedding_dimension_b,
            initializer=_get_initializer(embedding_dimension_b, embedding_values_b))

        # Test that columns are reordered alphabetically.
        sequence_input_layer = sfc.SequenceFeatures(
            [embedding_column_b, embedding_column_a])
        input_layer, sequence_length = sequence_input_layer({
            'aaa': sparse_input_a, 'bbb': sparse_input_b, })

        test_case.evaluate(variables_lib.global_variables_initializer())
        weights = sequence_input_layer.weights
        test_case.assertCountEqual(
            ('sequence_features/aaa_embedding/embedding_weights:0',
             'sequence_features/bbb_embedding/embedding_weights:0'),
            tuple([v.name for v in weights]))
        test_case.assertAllEqual(embedding_values_a, test_case.evaluate(weights[0]))
        test_case.assertAllEqual(embedding_values_b, test_case.evaluate(weights[1]))
        test_case.assertAllEqual(expected_input_layer, test_case.evaluate(input_layer))
        test_case.assertAllEqual(expected_sequence_length, test_case.evaluate(sequence_length))

    def _embedding_column_with_non_sequence_categorical(self):
        """Tests that error is raised for non-sequence embedding column."""
        vocabulary_size = 3
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            indices=((0, 0), (1, 0), (1, 1)),
            values=(2, 0, 1),
            dense_shape=(2, 2))

        categorical_column_a = fc.categorical_column_with_identity(
            key='aaa', num_buckets=vocabulary_size)
        embedding_column_a = fc.embedding_column(
            categorical_column_a, dimension=2)

        with test_case.assertRaisesRegexp(
                ValueError,
                r'In embedding_column: aaa_embedding\. categorical_column must be of '
                r'type SequenceCategoricalColumn to use SequenceFeatures\.'):
            sequence_input_layer = sfc.SequenceFeatures([embedding_column_a])
            _, _ = sequence_input_layer({'aaa': sparse_input})

    def _shared_embedding_column(self):
        with ops.Graph().as_default():
            vocabulary_size = 3
            sparse_input_a = sparse_tensor.SparseTensorValue(
                # example 0, ids [2]
                # example 1, ids [0, 1]
                indices=((0, 0), (1, 0), (1, 1)),
                values=(2, 0, 1),
                dense_shape=(2, 2))
            sparse_input_b = sparse_tensor.SparseTensorValue(
                # example 0, ids [1]
                # example 1, ids [2, 0]
                indices=((0, 0), (1, 0), (1, 1)),
                values=(1, 2, 0),
                dense_shape=(2, 2))

            embedding_dimension = 2
            embedding_values = (
                (1., 2.),  # id 0
                (3., 4.),  # id 1
                (5., 6.)  # id 2
            )

            def _get_initializer(embedding_dimension, embedding_values):
                def _initializer(shape, dtype, partition_info=None):
                    test_case.assertAllEqual((vocabulary_size, embedding_dimension), shape)
                    test_case.assertEqual(dtypes.float32, dtype)
                    test_case.assertIsNone(partition_info)
                    return embedding_values

                return _initializer

            expected_input_layer = [
                # example 0, ids_a [2], ids_b [1]
                [[5., 6., 3., 4.], [0., 0., 0., 0.]],
                # example 1, ids_a [0, 1], ids_b [2, 0]
                [[1., 2., 5., 6.], [3., 4., 1., 2.]],
            ]
            expected_sequence_length = [1, 2]

            categorical_column_a = sfc.sequence_categorical_column_with_identity(
                key='aaa', num_buckets=vocabulary_size)
            categorical_column_b = sfc.sequence_categorical_column_with_identity(
                key='bbb', num_buckets=vocabulary_size)
            # Test that columns are reordered alphabetically.
            shared_embedding_columns = fc.shared_embedding_columns_v2(
                [categorical_column_b, categorical_column_a],
                dimension=embedding_dimension,
                initializer=_get_initializer(embedding_dimension, embedding_values))

            sequence_input_layer = sfc.SequenceFeatures(shared_embedding_columns)
            input_layer, sequence_length = sequence_input_layer({'aaa': sparse_input_a, 'bbb': sparse_input_b})

            global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            test_case.assertCountEqual(('aaa_bbb_shared_embedding:0',), tuple([v.name for v in global_vars]))
            with _initialized_session() as sess:
                test_case.assertAllEqual(embedding_values, global_vars[0].eval(session=sess))
                test_case.assertAllEqual(expected_input_layer, input_layer.eval(session=sess))
                test_case.assertAllEqual(expected_sequence_length, sequence_length.eval(session=sess))

    def _shared_embedding_column_with_non_sequence_categorical(self):
        """Tests that error is raised for non-sequence shared embedding column."""
        vocabulary_size = 3
        sparse_input_a = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            indices=((0, 0), (1, 0), (1, 1)),
            values=(2, 0, 1),
            dense_shape=(2, 2))
        sparse_input_b = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            indices=((0, 0), (1, 0), (1, 1)),
            values=(2, 0, 1),
            dense_shape=(2, 2))

        categorical_column_a = fc.categorical_column_with_identity(key='aaa', num_buckets=vocabulary_size)
        categorical_column_b = fc.categorical_column_with_identity(key='bbb', num_buckets=vocabulary_size)
        shared_embedding_columns = fc.shared_embedding_columns_v2([categorical_column_a, categorical_column_b],
                                                                  dimension=2)

        with test_case.assertRaisesRegexp(
                ValueError,
                r'In embedding_column: aaa_shared_embedding\. categorical_column must '
                r'be of type SequenceCategoricalColumn to use SequenceFeatures\.'):
            sequence_input_layer = sfc.SequenceFeatures(shared_embedding_columns)
            _, _ = sequence_input_layer({'aaa': sparse_input_a, 'bbb': sparse_input_b})

    def _indicator_column(self, sparse_input_args_a, sparse_input_args_b, expected_input_layer,
                          expected_sequence_length):
        sparse_input_a = sparse_tensor.SparseTensorValue(**sparse_input_args_a)
        sparse_input_b = sparse_tensor.SparseTensorValue(**sparse_input_args_b)

        vocabulary_size_a = 3
        vocabulary_size_b = 2

        categorical_column_a = sfc.sequence_categorical_column_with_identity(
            key='aaa', num_buckets=vocabulary_size_a)
        indicator_column_a = fc.indicator_column(categorical_column_a)
        categorical_column_b = sfc.sequence_categorical_column_with_identity(
            key='bbb', num_buckets=vocabulary_size_b)
        indicator_column_b = fc.indicator_column(categorical_column_b)
        # Test that columns are reordered alphabetically.
        sequence_input_layer = sfc.SequenceFeatures(
            [indicator_column_b, indicator_column_a])
        input_layer, sequence_length = sequence_input_layer({
            'aaa': sparse_input_a, 'bbb': sparse_input_b})

        test_case.assertAllEqual(expected_input_layer, test_case.evaluate(input_layer))
        test_case.assertAllEqual(expected_sequence_length, test_case.evaluate(sequence_length))

    def _indicator_column_with_non_sequence_categorical(self):
        """Tests that error is raised for non-sequence categorical column."""
        vocabulary_size = 3
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            indices=((0, 0), (1, 0), (1, 1)),
            values=(2, 0, 1),
            dense_shape=(2, 2))

        categorical_column_a = fc.categorical_column_with_identity(
            key='aaa', num_buckets=vocabulary_size)
        indicator_column_a = fc.indicator_column(categorical_column_a)

        with test_case.assertRaisesRegexp(
                ValueError,
                r'In indicator_column: aaa_indicator\. categorical_column must be of '
                r'type SequenceCategoricalColumn to use SequenceFeatures\.'):
            sequence_input_layer = sfc.SequenceFeatures([indicator_column_a])
            _, _ = sequence_input_layer({'aaa': sparse_input})

    def _numeric_column(self, sparse_input_args, expected_input_layer, expected_sequence_length):
        sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)

        numeric_column = sfc.sequence_numeric_column('aaa')

        sequence_input_layer = sfc.SequenceFeatures([numeric_column])
        input_layer, sequence_length = sequence_input_layer({'aaa': sparse_input})

        test_case.assertAllEqual(expected_input_layer, test_case.evaluate(input_layer))
        test_case.assertAllEqual(
            expected_sequence_length, test_case.evaluate(sequence_length))

    def _numeric_column_multi_dim(self, sparse_input_args, expected_input_layer, expected_sequence_length):
        """Tests SequenceFeatures for multi-dimensional numeric_column."""
        sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)

        numeric_column = sfc.sequence_numeric_column('aaa', shape=(2, 2))

        sequence_input_layer = sfc.SequenceFeatures([numeric_column])
        input_layer, sequence_length = sequence_input_layer({'aaa': sparse_input})

        test_case.assertAllEqual(expected_input_layer, test_case.evaluate(input_layer))
        test_case.assertAllEqual(expected_sequence_length, test_case.evaluate(sequence_length))

    def _sequence_length_not_equal(self):
        """Tests that an error is raised when sequence lengths are not equal."""
        # Input a with sequence_length = [2, 1]
        sparse_input_a = sparse_tensor.SparseTensorValue(
            indices=((0, 0), (0, 1), (1, 0)),
            values=(0., 1., 10.),
            dense_shape=(2, 2))
        # Input b with sequence_length = [1, 1]
        sparse_input_b = sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0)),
            values=(1., 10.),
            dense_shape=(2, 2))
        numeric_column_a = sfc.sequence_numeric_column('aaa')
        numeric_column_b = sfc.sequence_numeric_column('bbb')

        sequence_input_layer = sfc.SequenceFeatures(
            [numeric_column_a, numeric_column_b])

        with test_case.assertRaisesRegexp(errors.InvalidArgumentError, r'Condition x == y did not hold.*'):
            _, sequence_length = sequence_input_layer({'aaa': sparse_input_a, 'bbb': sparse_input_b})
            test_case.evaluate(sequence_length)

    def _static_shape_from_tensors_numeric(self, sparse_input_args, expected_shape):
        """Tests that we return a known static shape when we have one."""
        sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)
        numeric_column = sfc.sequence_numeric_column('aaa', shape=(2, 2))

        sequence_input_layer = sfc.SequenceFeatures([numeric_column])
        input_layer, _ = sequence_input_layer({'aaa': sparse_input})
        shape = input_layer.get_shape()
        test_case.assertEqual(shape, expected_shape)

    def _static_shape_from_tensors_indicator(self, sparse_input_args, expected_shape):
        """Tests that we return a known static shape when we have one."""
        sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)
        categorical_column = sfc.sequence_categorical_column_with_identity(key='aaa', num_buckets=3)
        indicator_column = fc.indicator_column(categorical_column)

        sequence_input_layer = sfc.SequenceFeatures([indicator_column])
        input_layer, _ = sequence_input_layer({'aaa': sparse_input})
        shape = input_layer.get_shape()
        test_case.assertEqual(shape, expected_shape)

    def _compute_output_shape(self):
        price1 = sfc.sequence_numeric_column('price1', shape=2)
        price2 = sfc.sequence_numeric_column('price2')
        features = {
            'price1': sparse_tensor.SparseTensor(
                indices=[[0, 0, 0], [0, 0, 1],
                         [0, 1, 0], [0, 1, 1],
                         [1, 0, 0], [1, 0, 1],
                         [2, 0, 0], [2, 0, 1],
                         [3, 0, 0], [3, 0, 1]],
                values=[0., 1., 10., 11., 100., 101., 200., 201., 300., 301.],
                dense_shape=(4, 3, 2)),
            'price2': sparse_tensor.SparseTensor(
                indices=[[0, 0],
                         [0, 1],
                         [1, 0],
                         [2, 0],
                         [3, 0]],
                values=[10., 11., 20., 30., 40.],
                dense_shape=(4, 3))}
        sequence_features = sfc.SequenceFeatures([price1, price2])
        seq_input, seq_len = sequence_features(features)
        test_case.assertEqual(sequence_features.compute_output_shape((None, None)), (None, None, 3))
        test_case.evaluate(variables_lib.global_variables_initializer())
        test_case.evaluate(lookup_ops.tables_initializer())

        test_case.assertAllClose([[[0., 1., 10.], [10., 11., 11.], [0., 0., 0.]],
                                  [[100., 101., 20.], [0., 0., 0.], [0., 0., 0.]],
                                  [[200., 201., 30.], [0., 0., 0.], [0., 0., 0.]],
                                  [[300., 301., 40.], [0., 0., 0.], [0., 0., 0.]]],
                                 test_case.evaluate(seq_input))
        test_case.assertAllClose([2, 1, 1, 1], test_case.evaluate(seq_len))


class ConcatenateContextInputTes:
    """Tests the utility fn concatenate_context_input."""

    def run(self):
        params = [
            {'testcase_name': 'rank_lt_3',
             'seq_input_arg': np.arange(100).reshape(10, 10)},
            {'testcase_name': 'rank_gt_3',
             'seq_input_arg': np.arange(100).reshape((5, 5, 2, 2))}
        ]
        for param in params:
            self._sequence_input_throws_error(seq_input_arg=param['seq_input_arg'])
        params = [
            {'testcase_name': 'rank_lt_2',
             'context_input_arg': np.arange(100)},
            {'testcase_name': 'rank_gt_2',
             'context_input_arg': np.arange(100).reshape((5, 5, 4))}
        ]
        for param in params:
            self._context_input_throws_error(context_input_arg=param['context_input_arg'])

    def _concatenate_context_input(self):
        seq_input = ops.convert_to_tensor(np.arange(12).reshape((2, 3, 2)))
        context_input = ops.convert_to_tensor(np.arange(10).reshape(2, 5))
        seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
        context_input = math_ops.cast(context_input, dtype=dtypes.float32)
        input_layer = sfc.concatenate_context_input(context_input, seq_input)

        expected = np.array([
            [[0, 1, 0, 1, 2, 3, 4], [2, 3, 0, 1, 2, 3, 4], [4, 5, 0, 1, 2, 3, 4]],
            [[6, 7, 5, 6, 7, 8, 9], [8, 9, 5, 6, 7, 8, 9], [10, 11, 5, 6, 7, 8, 9]]
        ], dtype=np.float32)
        output = test_case.evaluate(input_layer)
        test_case.assertAllEqual(expected, output)

    def _sequence_input_throws_error(self, seq_input_arg):
        seq_input = ops.convert_to_tensor(seq_input_arg)
        context_input = ops.convert_to_tensor(np.arange(100).reshape(10, 10))
        seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
        context_input = math_ops.cast(context_input, dtype=dtypes.float32)
        with test_case.assertRaisesRegexp(ValueError, 'sequence_input must have rank 3'):
            sfc.concatenate_context_input(context_input, seq_input)

    def _context_input_throws_error(self, context_input_arg):
        context_input = ops.convert_to_tensor(context_input_arg)
        seq_input = ops.convert_to_tensor(np.arange(100).reshape((5, 5, 4)))
        seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
        context_input = math_ops.cast(context_input, dtype=dtypes.float32)
        with test_case.assertRaisesRegexp(ValueError, 'context_input must have rank 2'):
            sfc.concatenate_context_input(context_input, seq_input)

    def _integer_seq_input_throws_error(self):
        seq_input = ops.convert_to_tensor(np.arange(100).reshape((5, 5, 4)))
        context_input = ops.convert_to_tensor(np.arange(100).reshape(10, 10))
        context_input = math_ops.cast(context_input, dtype=dtypes.float32)
        with test_case.assertRaisesRegexp(TypeError, 'sequence_input must have dtype float32'):
            sfc.concatenate_context_input(context_input, seq_input)

    def _integer_context_input_throws_error(self):
        seq_input = ops.convert_to_tensor(np.arange(100).reshape([5, 5, 4]))
        context_input = ops.convert_to_tensor(np.arange(100).reshape(10, 10))
        seq_input = math_ops.cast(seq_input, dtype=dtypes.float32)
        with test_case.assertRaisesRegexp(TypeError, 'context_input must have dtype float32'):
            sfc.concatenate_context_input(context_input, seq_input)


class DenseFeaturesTes:
    """Tests DenseFeatures with sequence feature columns."""

    def _get_categorical(self):
        """Tests that error is raised for sequence embedding column."""
        vocabulary_size = 3
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            indices=((0, 0), (1, 0), (1, 1)),
            values=(2, 0, 1),
            dense_shape=(2, 2))

        categorical_column_a = sfc.sequence_categorical_column_with_identity(key='aaa', num_buckets=vocabulary_size)
        return categorical_column_a, sparse_input

    def _embedding_column(self):
        """Tests that error is raised for sequence embedding column."""
        categorical_column_a, sparse_input = self._get_categorical()
        embedding_column_a = fc.embedding_column(categorical_column_a, dimension=2)

        with test_case.assertRaisesRegexp(
                ValueError,
                r'In embedding_column: aaa_embedding\. categorical_column must not be '
                r'of type SequenceCategoricalColumn\.'):
            input_layer = dense_features.DenseFeatures([embedding_column_a])
            _ = input_layer({'aaa': sparse_input})

    def _indicator_column(self):
        """Tests that error is raised for sequence indicator column."""
        categorical_column_a, sparse_input = self._get_categorical()
        indicator_column_a = fc.indicator_column(categorical_column_a)

        with test_case.assertRaisesRegexp(
                ValueError,
                r'In indicator_column: aaa_indicator\. categorical_column must not be '
                r'of type SequenceCategoricalColumn\.'):
            input_layer = dense_features.DenseFeatures([indicator_column_a])
            _ = input_layer({'aaa': sparse_input})

    def run(self):
        self._embedding_column()
        self._indicator_column()


class SequenceCategoricalColumnWithIdentityTes:
    def _get_sparse_tensors(self, inputs_args, expected_args):
        inputs = sparse_tensor.SparseTensorValue(**inputs_args)
        expected = sparse_tensor.SparseTensorValue(**expected_args)
        column = sfc.sequence_categorical_column_with_identity('aaa', num_buckets=9)

        id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

        test_case.assertIsNone(id_weight_pair.weight_tensor)
        actual = test_case.evaluate(id_weight_pair.id_tensor)

        print(actual)
        _assert_sparse_tensor_value(expected, actual)

    def test_serialization(self):
        """Tests that column can be serialized."""
        parent = sfc.sequence_categorical_column_with_identity('animal', num_buckets=4)
        animal = fc.indicator_column(parent)

        config = animal.get_config()
        test_case.assertEqual(
            {
                'categorical_column': {
                    'class_name': 'SequenceCategoricalColumn',
                    'config': {
                        'categorical_column': {
                            'class_name': 'IdentityCategoricalColumn',
                            'config': {
                                'default_value': None,
                                'key': 'animal',
                                'number_buckets': 4
                            }
                        }
                    }
                }
            }, config)

        new_animal = fc.IndicatorColumn.from_config(config)
        test_case.assertEqual(animal, new_animal)
        test_case.assertIsNot(parent, new_animal.categorical_column)

        new_animal = fc.IndicatorColumn.from_config(config,
                                                    columns_by_name={
                                                        serialization._column_name_with_class_name(parent): parent
                                                    })
        test_case.assertEqual(animal, new_animal)
        test_case.assertIs(parent, new_animal.categorical_column)

    def run(self):
        # 2D
        self._get_sparse_tensors(
            inputs_args={
                'indices': ((0, 0), (1, 0), (1, 1)),
                'values': (1, 2, 0),
                'dense_shape': (2, 2)},
            expected_args={
                'indices': ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
                'values': np.array((1, 2, 0), dtype=np.int64),
                'dense_shape': (2, 2, 1)})
        # 3D
        self._get_sparse_tensors(
            inputs_args={
                'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
                'values': (6, 7, 8),
                'dense_shape': (2, 2, 2)},
            expected_args={
                'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
                'values': np.array((6, 7, 8), dtype=np.int64),
                'dense_shape': (2, 2, 2)})


class SequenceCategoricalColumnWithHashBucketTes:

    def _get_sparse_tensors(self, inputs_args, expected_args):
        inputs = sparse_tensor.SparseTensorValue(**inputs_args)
        expected = sparse_tensor.SparseTensorValue(**expected_args)
        column = sfc.sequence_categorical_column_with_hash_bucket('aaa', hash_bucket_size=10)

        id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

        test_case.assertIsNone(id_weight_pair.weight_tensor)
        actual = test_case.evaluate(id_weight_pair.id_tensor)
        _assert_sparse_tensor_indices_shape(expected, actual)

    def run(self):
        # 2D
        self._get_sparse_tensors(
            inputs_args={
                'indices': ((0, 0), (1, 0), (1, 1)),
                'values': ('omar', 'stringer', 'marlo'),
                'dense_shape': (2, 2)},
            expected_args={
                'indices': ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
                # Ignored to avoid hash dependence in test.
                'values': np.array((0, 0, 0), dtype=np.int64),
                'dense_shape': (2, 2, 1)})
        # 3D
        self._get_sparse_tensors(
            inputs_args={
                'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
                'values': ('omar', 'stringer', 'marlo'),
                'dense_shape': (2, 2, 2)},
            expected_args={
                'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
                # Ignored to avoid hash dependence in test.
                'values': np.array((0, 0, 0), dtype=np.int64),
                'dense_shape': (2, 2, 2)})


class SequenceCategoricalColumnWithVocabularyListTes:

    def _get_sparse_tensors(self, inputs_args, expected_args):
        inputs = sparse_tensor.SparseTensorValue(**inputs_args)
        expected = sparse_tensor.SparseTensorValue(**expected_args)
        column = sfc.sequence_categorical_column_with_vocabulary_list(
            key='aaa',
            vocabulary_list=('omar', 'stringer', 'marlo'))

        id_weight_pair = _get_sparse_tensors(column, {'aaa': inputs})

        test_case.assertIsNone(id_weight_pair.weight_tensor)
        test_case.evaluate(variables_lib.global_variables_initializer())
        test_case.evaluate(lookup_ops.tables_initializer())
        actual = test_case.evaluate(id_weight_pair.id_tensor)
        _assert_sparse_tensor_value(expected, actual)

    def run(self):
        self._get_sparse_tensors(
            inputs_args={
                'indices': ((0, 0), (1, 0), (1, 1)),
                'values': ('marlo', 'skywalker', 'omar'),
                'dense_shape': (2, 2)},
            expected_args={
                'indices': ((0, 0, 0), (1, 0, 0), (1, 1, 0)),
                'values': np.array((2, -1, 0), dtype=np.int64),
                'dense_shape': (2, 2, 1)}
        )
        self._get_sparse_tensors(
            inputs_args={
                'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
                'values': ('omar', 'skywalker', 'marlo'),
                'dense_shape': (2, 2, 2)},
            expected_args={
                'indices': ((0, 0, 2), (1, 0, 0), (1, 2, 0)),
                'values': np.array((0, -1, 2), dtype=np.int64),
                'dense_shape': (2, 2, 2)})


class SequenceEmbeddingColumnTes:

    def _get_sequence_dense_tensor(self, inputs_args, expected):
        inputs = sparse_tensor.SparseTensorValue(**inputs_args)
        vocabulary_size = 3
        embedding_dimension = 2

        categorical_column = sfc.sequence_categorical_column_with_identity(key='aaa', num_buckets=vocabulary_size)
        embedding_column = fc.embedding_column(categorical_column, dimension=embedding_dimension, )
        embedding_lookup, _, state_manager = _get_sequence_dense_tensor_state(embedding_column, {'aaa': inputs})

        embedding_column2 = fc.embedding_column(categorical_column, dimension=embedding_dimension, )
        embedding_lookup2, _, state_manager = _get_sequence_dense_tensor_state(embedding_column2, {'aaa': inputs})

        actual = test_case.evaluate(embedding_lookup)
        actual2 = test_case.evaluate(embedding_lookup2)

        print(actual)
        print(actual2)
        print("\n\n\n\n")

    def _sequence_length(self, inputs_args, expected_sequence_length):
        inputs = sparse_tensor.SparseTensorValue(**inputs_args)
        vocabulary_size = 3

        categorical_column = sfc.sequence_categorical_column_with_identity(key='aaa', num_buckets=vocabulary_size)
        embedding_column = fc.embedding_column(categorical_column, dimension=2)

        _, sequence_length, _ = _get_sequence_dense_tensor_state(embedding_column, {'aaa': inputs})

        sequence_length = test_case.evaluate(sequence_length)
        test_case.assertAllEqual(expected_sequence_length, sequence_length)
        test_case.assertEqual(np.int64, sequence_length.dtype)

    def _sequence_length_with_empty_rows(self):
        """Tests _sequence_length when some examples do not have ids."""
        vocabulary_size = 3
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids []
            # example 1, ids [2]
            # example 2, ids [0, 1]
            # example 3, ids []
            # example 4, ids [1]
            # example 5, ids []
            indices=((1, 0), (2, 0), (2, 1), (4, 0)),
            values=(2, 0, 1, 1),
            dense_shape=(6, 2))
        expected_sequence_length = [0, 1, 2, 0, 1, 0]

        categorical_column = sfc.sequence_categorical_column_with_identity(key='aaa', num_buckets=vocabulary_size)
        embedding_column = fc.embedding_column(categorical_column, dimension=2)

        _, sequence_length, _ = _get_sequence_dense_tensor_state(embedding_column, {'aaa': sparse_input})

        test_case.assertAllEqual(expected_sequence_length, test_case.evaluate(sequence_length))

    def run(self):
        params = [
            {'testcase_name': '2D',
             'inputs_args': {
                 # example 0, ids [2]
                 # example 1, ids [0, 1]
                 # example 2, ids []
                 # example 3, ids [1]
                 'indices': ((0, 0), (1, 0), (1, 1), (3, 0)),
                 'values': (2, 0, 1, 1),
                 'dense_shape': (4, 2)},
             'expected': [
                 # example 0, ids [2]
                 [[7., 11.], [0., 0.]],
                 # example 1, ids [0, 1]
                 [[1., 2.], [3., 5.]],
                 # example 2, ids []
                 [[0., 0.], [0., 0.]],
                 # example 3, ids [1]
                 [[3., 5.], [0., 0.]]]},
            {'testcase_name': '3D',
             'inputs_args': {
                 # example 0, ids [[2]]
                 # example 1, ids [[0, 1], [2]]
                 # example 2, ids []
                 # example 3, ids [[1], [0, 2]]
                 'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0),
                             (3, 0, 0), (3, 1, 0), (3, 1, 1)),
                 'values': (2, 0, 1, 2, 1, 0, 2),
                 'dense_shape': (4, 2, 2)},
             'expected': [
                 # example 0, ids [[2]]
                 [[7., 11.], [0., 0.]],
                 # example 1, ids [[0, 1], [2]]
                 [[2, 3.5], [7., 11.]],
                 # example 2, ids []
                 [[0., 0.], [0., 0.]],
                 # example 3, ids [[1], [0, 2]]
                 [[3., 5.], [4., 6.5]]]}
        ]
        for param in params:
            self._get_sequence_dense_tensor(inputs_args=param['inputs_args'], expected=param['expected'])
        params = [
            {'testcase_name': '2D',
             'inputs_args': {
                 # example 0, ids [2]
                 # example 1, ids [0, 1]
                 'indices': ((0, 0), (1, 0), (1, 1)),
                 'values': (2, 0, 1),
                 'dense_shape': (2, 2)},
             'expected_sequence_length': [1, 2]},
            {'testcase_name': '3D',
             'inputs_args': {
                 # example 0, ids [[2]]
                 # example 1, ids [[0, 1], [2]]
                 'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
                 'values': (2, 0, 1, 2),
                 'dense_shape': (2, 2, 2)},
             'expected_sequence_length': [1, 2]}]

        for param in params:
            self._sequence_length(inputs_args=param['inputs_args'],
                                  expected_sequence_length=param['expected_sequence_length'])


class SequenceSharedEmbeddingColumnTes:

    def _get_sequence_dense_tensor(self):
        vocabulary_size = 3
        embedding_dimension = 2
        embedding_values = (
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.)  # id 2
        )

        def _initializer(shape, dtype, partition_info=None):
            test_case.assertAllEqual((vocabulary_size, embedding_dimension), shape)
            test_case.assertEqual(dtypes.float32, dtype)
            test_case.assertIsNone(partition_info)
            return embedding_values

        sparse_input_a = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=(2, 0, 1, 1),
            dense_shape=(4, 2))
        sparse_input_b = sparse_tensor.SparseTensorValue(
            # example 0, ids [1]
            # example 1, ids [0, 2]
            # example 2, ids [0]
            # example 3, ids []
            indices=((0, 0), (1, 0), (1, 1), (2, 0)),
            values=(1, 0, 2, 0),
            dense_shape=(4, 2))

        expected_lookups_a = [
            # example 0, ids [2]
            [[7., 11.], [0., 0.]],
            # example 1, ids [0, 1]
            [[1., 2.], [3., 5.]],
            # example 2, ids []
            [[0., 0.], [0., 0.]],
            # example 3, ids [1]
            [[3., 5.], [0., 0.]],
        ]

        expected_lookups_b = [
            # example 0, ids [1]
            [[3., 5.], [0., 0.]],
            # example 1, ids [0, 2]
            [[1., 2.], [7., 11.]],
            # example 2, ids [0]
            [[1., 2.], [0., 0.]],
            # example 3, ids []
            [[0., 0.], [0., 0.]],
        ]

        categorical_column_a = sfc.sequence_categorical_column_with_identity(key='aaa', num_buckets=vocabulary_size)
        categorical_column_b = sfc.sequence_categorical_column_with_identity(key='bbb', num_buckets=vocabulary_size)
        shared_embedding_columns = fc.shared_embedding_columns_v2(
            [categorical_column_a, categorical_column_b],
            dimension=embedding_dimension,
            initializer=_initializer)

        embedding_lookup_a = _get_sequence_dense_tensor(shared_embedding_columns[0], {'aaa': sparse_input_a})[0]
        embedding_lookup_b = _get_sequence_dense_tensor(shared_embedding_columns[1], {'bbb': sparse_input_b})[0]

        test_case.evaluate(variables_lib.global_variables_initializer())
        global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
        test_case.assertItemsEqual(('aaa_bbb_shared_embedding:0',), tuple([v.name for v in global_vars]))
        test_case.assertAllEqual(embedding_values, test_case.evaluate(global_vars[0]))
        test_case.assertAllEqual(expected_lookups_a, test_case.evaluate(embedding_lookup_a))
        test_case.assertAllEqual(expected_lookups_b, test_case.evaluate(embedding_lookup_b))

    def _sequence_length(self):
        with ops.Graph().as_default():
            vocabulary_size = 3

            sparse_input_a = sparse_tensor.SparseTensorValue(
                # example 0, ids [2]
                # example 1, ids [0, 1]
                indices=((0, 0), (1, 0), (1, 1)),
                values=(2, 0, 1),
                dense_shape=(2, 2))
            expected_sequence_length_a = [1, 2]
            categorical_column_a = sfc.sequence_categorical_column_with_identity(
                key='aaa', num_buckets=vocabulary_size)

            sparse_input_b = sparse_tensor.SparseTensorValue(
                # example 0, ids [0, 2]
                # example 1, ids [1]
                indices=((0, 0), (0, 1), (1, 0)),
                values=(0, 2, 1),
                dense_shape=(2, 2))
            expected_sequence_length_b = [2, 1]
            categorical_column_b = sfc.sequence_categorical_column_with_identity(key='bbb', num_buckets=vocabulary_size)
            shared_embedding_columns = fc.shared_embedding_columns_v2([categorical_column_a, categorical_column_b],
                                                                      dimension=2)

            sequence_length_a = _get_sequence_dense_tensor(shared_embedding_columns[0], {'aaa': sparse_input_a})[1]
            sequence_length_b = _get_sequence_dense_tensor(shared_embedding_columns[1], {'bbb': sparse_input_b})[1]

            with _initialized_session() as sess:
                sequence_length_a = sess.run(sequence_length_a)
                test_case.assertAllEqual(expected_sequence_length_a, sequence_length_a)
                test_case.assertEqual(np.int64, sequence_length_a.dtype)
                sequence_length_b = sess.run(sequence_length_b)
                test_case.assertAllEqual(expected_sequence_length_b, sequence_length_b)
                test_case.assertEqual(np.int64, sequence_length_b.dtype)

    def _sequence_length_with_empty_rows(self):
        """Tests _sequence_length when some examples do not have ids."""
        with ops.Graph().as_default():
            vocabulary_size = 3
            sparse_input_a = sparse_tensor.SparseTensorValue(
                # example 0, ids []
                # example 1, ids [2]
                # example 2, ids [0, 1]
                # example 3, ids []
                # example 4, ids [1]
                # example 5, ids []
                indices=((1, 0), (2, 0), (2, 1), (4, 0)),
                values=(2, 0, 1, 1),
                dense_shape=(6, 2))
            expected_sequence_length_a = [0, 1, 2, 0, 1, 0]
            categorical_column_a = sfc.sequence_categorical_column_with_identity(
                key='aaa', num_buckets=vocabulary_size)

            sparse_input_b = sparse_tensor.SparseTensorValue(
                # example 0, ids [2]
                # example 1, ids []
                # example 2, ids []
                # example 3, ids []
                # example 4, ids [1]
                # example 5, ids [0, 1]
                indices=((0, 0), (4, 0), (5, 0), (5, 1)),
                values=(2, 1, 0, 1),
                dense_shape=(6, 2))
            expected_sequence_length_b = [1, 0, 0, 0, 1, 2]
            categorical_column_b = sfc.sequence_categorical_column_with_identity(key='bbb', num_buckets=vocabulary_size)

            shared_embedding_columns = fc.shared_embedding_columns_v2([categorical_column_a, categorical_column_b],
                                                                      dimension=2)

            sequence_length_a = _get_sequence_dense_tensor(shared_embedding_columns[0], {'aaa': sparse_input_a})[1]
            sequence_length_b = _get_sequence_dense_tensor(shared_embedding_columns[1], {'bbb': sparse_input_b})[1]

            with _initialized_session() as sess:
                test_case.assertAllEqual(expected_sequence_length_a, sequence_length_a.eval(session=sess))
                test_case.assertAllEqual(expected_sequence_length_b, sequence_length_b.eval(session=sess))

    def run(self):
        self._get_sequence_dense_tensor()
        self._sequence_length()
        self._sequence_length_with_empty_rows()


class SequenceSharedEmbeddingColumnTes2:

    def _get_sequence_dense_tensor(self):
        vocabulary_size = 3
        embedding_dimension = 2

        sparse_input_a = sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (3, 0)),
            values=(2, 0, 1, 1),
            dense_shape=(4, 2))
        sparse_input_b = sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (2, 0)),
            values=(1, 0, 2, 0),
            dense_shape=(4, 2))

        categorical_column_a = sfc.sequence_categorical_column_with_identity(key='aaa', num_buckets=vocabulary_size)
        categorical_column_b = sfc.sequence_categorical_column_with_identity(key='bbb', num_buckets=vocabulary_size)
        categorical_column_c = sfc.sequence_categorical_column_with_identity(key='aaa', num_buckets=vocabulary_size)
        categorical_column_d = sfc.sequence_categorical_column_with_identity(key='bbb', num_buckets=vocabulary_size)
        with tf.compat.v1.variable_scope("share", reuse=tf.compat.v1.AUTO_REUSE):
            shared_embedding_columns = fc.shared_embedding_columns_v2([categorical_column_a],
                                                                      dimension=embedding_dimension,
                                                                      shared_embedding_collection_name="a")
            shared_embedding_columns2 = fc.shared_embedding_columns_v2([categorical_column_c, categorical_column_d],
                                                                       dimension=embedding_dimension,
                                                                       shared_embedding_collection_name='a')

            embedding_lookup_a = sfc.SequenceFeatures(shared_embedding_columns[0])({'aaa': sparse_input_a})[0]
            # embedding_lookup_b = sfc.SequenceFeatures(shared_embedding_columns[1])({'bbb': sparse_input_b})[0]
            embedding_lookup_c = sfc.SequenceFeatures(shared_embedding_columns2[0])({'aaa': sparse_input_a})[0]
            embedding_lookup_d = _get_sequence_dense_tensor(shared_embedding_columns2[1], {'bbb': sparse_input_b})[0]

        test_case.evaluate(variables_lib.global_variables_initializer())

        actual1 = test_case.evaluate(embedding_lookup_a)
        actual2 = test_case.evaluate(embedding_lookup_c)

        print(actual1)
        print(actual2)

    def _sequence_length(self):
        with ops.Graph().as_default():
            vocabulary_size = 3

            sparse_input_a = sparse_tensor.SparseTensorValue(
                # example 0, ids [2]
                # example 1, ids [0, 1]
                indices=((0, 0), (1, 0), (1, 1)),
                values=(2, 0, 1),
                dense_shape=(2, 2))
            expected_sequence_length_a = [1, 2]
            categorical_column_a = sfc.sequence_categorical_column_with_identity(
                key='aaa', num_buckets=vocabulary_size)

            sparse_input_b = sparse_tensor.SparseTensorValue(
                # example 0, ids [0, 2]
                # example 1, ids [1]
                indices=((0, 0), (0, 1), (1, 0)),
                values=(0, 2, 1),
                dense_shape=(2, 2))
            expected_sequence_length_b = [2, 1]
            categorical_column_b = sfc.sequence_categorical_column_with_identity(key='bbb', num_buckets=vocabulary_size)
            shared_embedding_columns = fc.shared_embedding_columns_v2([categorical_column_a, categorical_column_b],
                                                                      dimension=2)

            sequence_length_a = _get_sequence_dense_tensor(shared_embedding_columns[0], {'aaa': sparse_input_a})[1]
            sequence_length_b = _get_sequence_dense_tensor(shared_embedding_columns[1], {'bbb': sparse_input_b})[1]

            with _initialized_session() as sess:
                sequence_length_a = sess.run(sequence_length_a)
                test_case.assertAllEqual(expected_sequence_length_a, sequence_length_a)
                test_case.assertEqual(np.int64, sequence_length_a.dtype)
                sequence_length_b = sess.run(sequence_length_b)
                test_case.assertAllEqual(expected_sequence_length_b, sequence_length_b)
                test_case.assertEqual(np.int64, sequence_length_b.dtype)

    def _sequence_length_with_empty_rows(self):
        """Tests _sequence_length when some examples do not have ids."""
        with ops.Graph().as_default():
            vocabulary_size = 3
            sparse_input_a = sparse_tensor.SparseTensorValue(
                # example 0, ids []
                # example 1, ids [2]
                # example 2, ids [0, 1]
                # example 3, ids []
                # example 4, ids [1]
                # example 5, ids []
                indices=((1, 0), (2, 0), (2, 1), (4, 0)),
                values=(2, 0, 1, 1),
                dense_shape=(6, 2))
            expected_sequence_length_a = [0, 1, 2, 0, 1, 0]
            categorical_column_a = sfc.sequence_categorical_column_with_identity(
                key='aaa', num_buckets=vocabulary_size)

            sparse_input_b = sparse_tensor.SparseTensorValue(
                # example 0, ids [2]
                # example 1, ids []
                # example 2, ids []
                # example 3, ids []
                # example 4, ids [1]
                # example 5, ids [0, 1]
                indices=((0, 0), (4, 0), (5, 0), (5, 1)),
                values=(2, 1, 0, 1),
                dense_shape=(6, 2))
            expected_sequence_length_b = [1, 0, 0, 0, 1, 2]
            categorical_column_b = sfc.sequence_categorical_column_with_identity(key='bbb', num_buckets=vocabulary_size)

            shared_embedding_columns = fc.shared_embedding_columns_v2([categorical_column_a, categorical_column_b],
                                                                      dimension=2)

            sequence_length_a = _get_sequence_dense_tensor(shared_embedding_columns[0], {'aaa': sparse_input_a})[1]
            sequence_length_b = _get_sequence_dense_tensor(shared_embedding_columns[1], {'bbb': sparse_input_b})[1]

            with _initialized_session() as sess:
                test_case.assertAllEqual(expected_sequence_length_a, sequence_length_a.eval(session=sess))
                test_case.assertAllEqual(expected_sequence_length_b, sequence_length_b.eval(session=sess))

    def run(self):
        self._get_sequence_dense_tensor()
        self._sequence_length()
        self._sequence_length_with_empty_rows()


class SequenceIndicatorColumnTes:
    def run(self):
        params = [
            {'testcase_name': '2D',
             'inputs_args': {
                 # example 0, ids [2]
                 # example 1, ids [0, 1]
                 # example 2, ids []
                 # example 3, ids [1]
                 'indices': ((0, 0), (1, 0), (1, 1), (3, 0)),
                 'values': (2, 0, 1, 1),
                 'dense_shape': (4, 2)},
             'expected': [
                 # example 0, ids [2]
                 [[0., 0., 1.], [0., 0., 0.]],
                 # example 1, ids [0, 1]
                 [[1., 0., 0.], [0., 1., 0.]],
                 # example 2, ids []
                 [[0., 0., 0.], [0., 0., 0.]],
                 # example 3, ids [1]
                 [[0., 1., 0.], [0., 0., 0.]]]},
            {'testcase_name': '3D',
             'inputs_args': {
                 # example 0, ids [[2]]
                 # example 1, ids [[0, 1], [2]]
                 # example 2, ids []
                 # example 3, ids [[1], [2, 2]]
                 'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0),
                             (3, 0, 0), (3, 1, 0), (3, 1, 1)),
                 'values': (2, 0, 1, 2, 1, 2, 2),
                 'dense_shape': (4, 2, 2)},
             'expected': [
                 # example 0, ids [[2]]
                 [[0., 0., 1.], [0., 0., 0.]],
                 # example 1, ids [[0, 1], [2]]
                 [[1., 1., 0.], [0., 0., 1.]],
                 # example 2, ids []
                 [[0., 0., 0.], [0., 0., 0.]],
                 # example 3, ids [[1], [2, 2]]
                 [[0., 1., 0.], [0., 0., 2.]]]}
        ]

        for param in params:
            self._get_sequence_dense_tensor(inputs_args=param['inputs_args'], expected=param['expected'])

        params = [
            {'testcase_name': '2D',
             'inputs_args': {
                 # example 0, ids [2]
                 # example 1, ids [0, 1]
                 'indices': ((0, 0), (1, 0), (1, 1)),
                 'values': (2, 0, 1),
                 'dense_shape': (2, 2)},
             'expected_sequence_length': [1, 2]},
            {'testcase_name': '3D',
             'inputs_args': {
                 # example 0, ids [[2]]
                 # example 1, ids [[0, 1], [2]]
                 'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
                 'values': (2, 0, 1, 2),
                 'dense_shape': (2, 2, 2)},
             'expected_sequence_length': [1, 2]}
        ]
        for param in params:
            self._sequence_length(inputs_args=param['inputs_args'],
                                  expected_sequence_length=param['expected_sequence_length'])

    def _get_sequence_dense_tensor(self, inputs_args, expected):
        inputs = sparse_tensor.SparseTensorValue(**inputs_args)
        vocabulary_size = 3

        categorical_column = sfc.sequence_categorical_column_with_identity(
            key='aaa', num_buckets=vocabulary_size)
        indicator_column = fc.indicator_column(categorical_column)

        indicator_tensor, _ = _get_sequence_dense_tensor(
            indicator_column, {'aaa': inputs})

        test_case.assertAllEqual(expected, test_case.evaluate(indicator_tensor))

    def _sequence_length(self, inputs_args, expected_sequence_length):
        inputs = sparse_tensor.SparseTensorValue(**inputs_args)
        vocabulary_size = 3

        categorical_column = sfc.sequence_categorical_column_with_identity(
            key='aaa', num_buckets=vocabulary_size)
        indicator_column = fc.indicator_column(categorical_column)

        _, sequence_length = _get_sequence_dense_tensor(
            indicator_column, {'aaa': inputs})

        sequence_length = test_case.evaluate(sequence_length)
        test_case.assertAllEqual(expected_sequence_length, sequence_length)
        test_case.assertEqual(np.int64, sequence_length.dtype)

    def _sequence_length_with_empty_rows(self):
        """Tests _sequence_length when some examples do not have ids."""
        vocabulary_size = 3
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids []
            # example 1, ids [2]
            # example 2, ids [0, 1]
            # example 3, ids []
            # example 4, ids [1]
            # example 5, ids []
            indices=((1, 0), (2, 0), (2, 1), (4, 0)),
            values=(2, 0, 1, 1),
            dense_shape=(6, 2))
        expected_sequence_length = [0, 1, 2, 0, 1, 0]

        categorical_column = sfc.sequence_categorical_column_with_identity(
            key='aaa', num_buckets=vocabulary_size)
        indicator_column = fc.indicator_column(categorical_column)

        _, sequence_length = _get_sequence_dense_tensor(
            indicator_column, {'aaa': sparse_input})

        test_case.assertAllEqual(expected_sequence_length, test_case.evaluate(sequence_length))


class SequenceNumericColumnTes:

    def run(self):
        params = [
            {'testcase_name': '2D',
             'inputs_args': {
                 # example 0, values [0., 1]
                 # example 1, [10.]
                 'indices': ((0, 0), (0, 1), (1, 0)),
                 'values': (0., 1., 10.),
                 'dense_shape': (2, 2)},
             'expected': [
                 [[0.], [1.]],
                 [[10.], [0.]]]},
            {'testcase_name': '3D',
             'inputs_args': {
                 # feature 0, ids [[20, 3], [5]]
                 # feature 1, ids [[3], [8]]
                 'indices': ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
                 'values': (20, 3, 5., 3., 8.),
                 'dense_shape': (2, 2, 2)},
             'expected': [
                 [[20.], [3.], [5.], [0.]],
                 [[3.], [0.], [8.], [0.]]]},
        ]

        for param in params:
            self._get_sequence_dense_tensor(inputs_args=param['inputs_args'], expected=param['expected'])

        params = [
            {'testcase_name': '2D',
             'sparse_input_args': {
                 # example 0, values [[[0., 1.],  [2., 3.]], [[4., 5.],  [6., 7.]]]
                 # example 1, [[[10., 11.],  [12., 13.]]]
                 'indices': ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                             (0, 7), (1, 0), (1, 1), (1, 2), (1, 3)),
                 'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
                 'dense_shape': (2, 8)},
             'expected_dense_tensor': [
                 [[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
                 [[[10., 11.], [12., 13.]], [[0., 0.], [0., 0.]]]]},
            {'testcase_name': '3D',
             'sparse_input_args': {
                 'indices': ((0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6),
                             (0, 1, 0), (0, 1, 2), (0, 1, 4), (0, 1, 6),
                             (1, 0, 0), (1, 0, 2), (1, 0, 4), (1, 0, 6)),
                 'values': (0., 1., 2., 3., 4., 5., 6., 7., 10., 11., 12., 13.),
                 'dense_shape': (2, 2, 8)},
             'expected_dense_tensor': [
                 [[[0., 0.], [1., 0.]], [[2., 0.], [3., 0.]],
                  [[4., 0.], [5., 0.]], [[6., 0.], [7., 0.]]],
                 [[[10., 0.], [11., 0.]], [[12., 0.], [13., 0.]],
                  [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]]]},
        ]
        for param in params:
            self._get_dense_tensor_multi_dim(sparse_input_args=param['sparse_input_args'],
                                             expected_dense_tensor=param['expected_dense_tensor'])

        params = [
            {'testcase_name': '2D',
             'inputs_args': {
                 # example 0, ids [2]
                 # example 1, ids [0, 1]
                 'indices': ((0, 0), (1, 0), (1, 1)),
                 'values': (2., 0., 1.),
                 'dense_shape': (2, 2)},
             'expected_sequence_length': [1, 2],
             'shape': (1,)},
            {'testcase_name': '3D',
             'inputs_args': {
                 # example 0, ids [[2]]
                 # example 1, ids [[0, 1], [2]]
                 'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
                 'values': (2., 0., 1., 2.),
                 'dense_shape': (2, 2, 2)},
             'expected_sequence_length': [1, 2],
             'shape': (1,)},
            {'testcase_name': '2D_with_shape',
             'inputs_args': {
                 # example 0, ids [2]
                 # example 1, ids [0, 1]
                 'indices': ((0, 0), (1, 0), (1, 1)),
                 'values': (2., 0., 1.),
                 'dense_shape': (2, 2)},
             'expected_sequence_length': [1, 1],
             'shape': (2,)},
            {'testcase_name': '3D_with_shape',
             'inputs_args': {
                 # example 0, ids [[2]]
                 # example 1, ids [[0, 1], [2]]
                 'indices': ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 0)),
                 'values': (2., 0., 1., 2.),
                 'dense_shape': (2, 2, 2)},
             'expected_sequence_length': [1, 2],
             'shape': (2,)},
        ]
        for param in params:
            self._sequence_length(inputs_args=param['inputs_args'],
                                  expected_sequence_length=param['expected_sequence_length'], shape=param['shape'])

    def test_defaults(self):
        a = sfc.sequence_numeric_column('aaa')
        test_case.assertEqual('aaa', a.key)
        test_case.assertEqual('aaa', a.name)
        test_case.assertEqual((1,), a.shape)
        test_case.assertEqual(0., a.default_value)
        test_case.assertEqual(dtypes.float32, a.dtype)
        test_case.assertIsNone(a.normalizer_fn)

    def test_shape_saved_as_tuple(self):
        a = sfc.sequence_numeric_column('aaa', shape=[1, 2])
        test_case.assertEqual((1, 2), a.shape)

    def test_shape_must_be_positive_integer(self):
        with test_case.assertRaisesRegexp(TypeError, 'shape dimensions must be integer'):
            sfc.sequence_numeric_column('aaa', shape=[1.0])

        with test_case.assertRaisesRegexp(ValueError, 'shape dimensions must be greater than 0'):
            sfc.sequence_numeric_column('aaa', shape=[0])

    def test_dtype_is_convertible_to_float(self):
        with test_case.assertRaisesRegexp(ValueError, 'dtype must be convertible to float'):
            sfc.sequence_numeric_column('aaa', dtype=dtypes.string)

    def test_normalizer_fn_must_be_callable(self):
        with test_case.assertRaisesRegexp(TypeError, 'must be a callable'):
            sfc.sequence_numeric_column('aaa', normalizer_fn='NotACallable')

    def _get_sequence_dense_tensor(self, inputs_args, expected):
        inputs = sparse_tensor.SparseTensorValue(**inputs_args)
        numeric_column = sfc.sequence_numeric_column('aaa')

        dense_tensor, _ = _get_sequence_dense_tensor(
            numeric_column, {'aaa': inputs})
        test_case.assertAllEqual(expected, test_case.evaluate(dense_tensor))

    def _get_sequence_dense_tensor_with_normalizer_fn(self):
        def _increment_two(input_sparse_tensor):
            return sparse_ops.sparse_add(
                input_sparse_tensor,
                sparse_tensor.SparseTensor(((0, 0), (1, 1)), (2.0, 2.0), (2, 2))
            )

        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, values [[0.], [1]]
            # example 1, [[10.]]
            indices=((0, 0), (0, 1), (1, 0)),
            values=(0., 1., 10.),
            dense_shape=(2, 2))

        # Before _increment_two:
        #   [[0.], [1.]],
        #   [[10.], [0.]],
        # After _increment_two:
        #   [[2.], [1.]],
        #   [[10.], [2.]],
        expected_dense_tensor = [
            [[2.], [1.]],
            [[10.], [2.]],
        ]
        numeric_column = sfc.sequence_numeric_column(
            'aaa', normalizer_fn=_increment_two)

        dense_tensor, _ = _get_sequence_dense_tensor(
            numeric_column, {'aaa': sparse_input})

        test_case.assertAllEqual(expected_dense_tensor, test_case.evaluate(dense_tensor))

    def _get_dense_tensor_multi_dim(self, sparse_input_args, expected_dense_tensor):
        """Tests get_sequence_dense_tensor for multi-dim numeric_column."""
        sparse_input = sparse_tensor.SparseTensorValue(**sparse_input_args)
        numeric_column = sfc.sequence_numeric_column('aaa', shape=(2, 2))

        dense_tensor, _ = _get_sequence_dense_tensor(
            numeric_column, {'aaa': sparse_input})

        test_case.assertAllEqual(expected_dense_tensor, test_case.evaluate(dense_tensor))

    def _sequence_length(self, inputs_args, expected_sequence_length, shape):
        inputs = sparse_tensor.SparseTensorValue(**inputs_args)
        numeric_column = sfc.sequence_numeric_column('aaa', shape=shape)

        _, sequence_length = _get_sequence_dense_tensor(
            numeric_column, {'aaa': inputs})

        sequence_length = test_case.evaluate(sequence_length)
        test_case.assertAllEqual(expected_sequence_length, sequence_length)
        test_case.assertEqual(np.int64, sequence_length.dtype)

    def test_sequence_length_with_empty_rows(self):
        """Tests _sequence_length when some examples do not have ids."""
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, values []
            # example 1, values [[0.], [1.]]
            # example 2, [[2.]]
            # example 3, values []
            # example 4, [[3.]]
            # example 5, values []
            indices=((1, 0), (1, 1), (2, 0), (4, 0)),
            values=(0., 1., 2., 3.),
            dense_shape=(6, 2))
        expected_sequence_length = [0, 2, 1, 0, 1, 0]
        numeric_column = sfc.sequence_numeric_column('aaa')

        _, sequence_length = _get_sequence_dense_tensor(
            numeric_column, {'aaa': sparse_input})

        test_case.assertAllEqual(expected_sequence_length, test_case.evaluate(sequence_length))

    def test_serialization(self):
        """Tests that column can be serialized."""

        def _custom_fn(input_tensor):
            return input_tensor + 42

        column = sfc.sequence_numeric_column(
            key='my-key', shape=(2,), default_value=3, dtype=dtypes.int32,
            normalizer_fn=_custom_fn)
        configs = serialization.serialize_feature_column(column)
        column = serialization.deserialize_feature_column(
            configs, custom_objects={_custom_fn.__name__: _custom_fn})
        test_case.assertEqual(column.key, 'my-key')
        test_case.assertEqual(column.shape, (2,))
        test_case.assertEqual(column.default_value, 3)
        test_case.assertEqual(column.normalizer_fn(3), 45)
        with test_case.assertRaisesRegex(ValueError, 'Instance: 0 is not a FeatureColumn'):
            serialization.serialize_feature_column(int())

    def test_parents(self):
        """Tests parents attribute of column."""
        column = sfc.sequence_numeric_column(key='my-key')
        test_case.assertEqual(column.parents, ['my-key'])


def example():
    URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
    dataframe = pd.read_csv(URL)
    dataframe.head()

    train, test = train_test_split(dataframe, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    import random
    list = ["a", 'b', 'c', 'd', 'e', 'f']

    def arr_c(n):
        res = []
        for i in range(0, n):
            random.shuffle(list)
            res.append(np.array(list[:4]))
        return res

    # 一种从 Pandas Dataframe 创建 tf.data 数据集的实用程序方法（utility method）
    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('target')
        dataframe['arr'] = arr_c(len(dataframe))
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        ds = ds.batch(batch_size)
        return ds

    batch_size = 5  # 小批量大小用于演示
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    # 我们将使用该批数据演示几种特征列
    example_batch = next(iter(train_ds))[0]

    from tensorflow.python.feature_column import sequence_feature_column as sfc

    thal_seq = feature_column.sequence_categorical_column_with_vocabulary_list('arr', ['a', 'b', 'c', 'd', 'e', 'f'])

    # thal_one_hot = feature_column.embedding_column(thal_seq, dimension=8)
    # thal_one_hot = feature_column.indicator_column(thal_seq)
    from notekeras.feature.feature_column_def import IndicatorColumnDef

    thal_one_hot = IndicatorColumnDef(thal_seq, size=15)

    def _get_sequence_dense_tensor_state(column, features):
        sequence_feature_layer = sfc.SequenceFeatures(column)
        sequence_input, sequence_length = sequence_feature_layer(features)
        return sequence_input, sequence_length

    print(example_batch)
    sequence_input, sequence_length = _get_sequence_dense_tensor_state(thal_one_hot, example_batch)

    sequence_input = tf.keras.backend.sum(sequence_input, axis=-1)
    print(sequence_input)

    from notekeras.layers.embedding import EmbeddingRet
    res = EmbeddingRet(input_dim=6,
                       output_dim=11,
                       mask_zero=True,
                       name='Token-Embedding')(sequence_input)

    print(res)


example()
# SequenceEmbeddingColumnTes().run()
# SequenceSharedEmbeddingColumnTes2().run()
#
#
# SequenceCategoricalColumnWithIdentityTes().run()
# SequenceCategoricalColumnWithHashBucketTes().run()
# SequenceCategoricalColumnWithVocabularyListTes().run()
