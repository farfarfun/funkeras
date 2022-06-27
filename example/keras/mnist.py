import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

train_ds = tfds.load("mnist", split=tfds.Split.TRAIN, batch_size=-1)
numpy_ds = tfds.as_numpy(train_ds)

mnist_train_ds = tf.data.Dataset.from_tensor_slices((numpy_ds["image"], numpy_ds["label"]))

mnist_train_ds = mnist_train_ds.shuffle(5000).batch(32).repeat(3)

model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(64),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(mnist_train_ds, epochs=2)
