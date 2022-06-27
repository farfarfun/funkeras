import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow_core.python.keras.utils import np_utils

mnist_builder = tfds.builder('iris')
mnist_builder.download_and_prepare()
ds = mnist_builder.as_dataset(split='train').batch(32).repeat(4)
print(ds)

model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(64),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3)
])


def generator_data():
    for data in ds:
        image = data['features'].numpy()
        label = data['label'].numpy()

        # image = image.reshape([32, 28, 28, 1])
        # label = label.reshape([32, 1])
        # print(label)

        label = np_utils.to_categorical(label, num_classes=3)
        print(image.shape, label.shape)

        yield image, label, [None]


model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.fit(generator_data(), steps_per_epoch=10)
model.fit(generator_data(), steps_per_epoch=10)
