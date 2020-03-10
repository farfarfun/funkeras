import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model

tfds.list_builders()
dataset, info = tfds.load("mnist", with_info=True)
inputs = Input((28, 28, 1), name="image")

first = Dense(128, activation="relu")(inputs)
second = Dropout(0.2)(first)
third = Dense(10, activation="softmax", name="label")(second)
model = Model(inputs=[inputs], outputs=[third])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(dataset['train'].batch(4096))
