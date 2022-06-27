import tensorflow_datasets as tfds

# tfds works in both Eager and Graph modes
# tf.compat.v1.enable_eager_execution()

# See available datasets
print(tfds.list_builders())

# Construct a tf.data.Dataset
ds_train = tfds.load(name="mnist", split="train", shuffle_files=True)

# Build your input pipeline
ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
for features in ds_train.take(1):
    image, label = features["image"], features["label"]

import keras

keras.constraints.serialize()
# https://github.com/notechats/notedrive.git
