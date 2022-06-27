from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.keras.utils import np_utils


def sequential_model_with_ds_input():
    columns = [fc.numeric_column('a')]
    model = keras.models.Sequential([
        fc.DenseFeatures(columns),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(20, activation='softmax')
    ])
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    y = np.random.randint(20, size=(100, 1))
    y = np_utils.to_categorical(y, num_classes=20)
    x = {'a': np.random.random((100, 1))}
    ds1 = dataset_ops.Dataset.from_tensor_slices(x)
    ds2 = dataset_ops.Dataset.from_tensor_slices(y)

    ds = dataset_ops.Dataset.zip((ds1, ds2)).batch(5)

    model.fit(ds, steps_per_epoch=1)

    model.evaluate(ds, steps=1)
    model.predict(ds, steps=1)


sequential_model_with_ds_input()
