import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from notekeras.layer.fm import FM
from sklearn.datasets import load_breast_cancer
from tensorflow import keras
from tensorflow.keras import Model, activations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Input, Layer
from tensorflow.keras.optimizers import Adam

data = load_breast_cancer()["data"]
target = load_breast_cancer()["target"]


print(target)
inputs = Input(shape=(30,))
out = FM(20)(inputs)
out = Dense(15, activation='sigmoid')(out)
out = Dense(1, activation='sigmoid')(out)

model = Model(inputs=inputs, outputs=out)
model.compile(loss='mse',
              optimizer='adam',
              metrics=['acc'])
model.summary()

h = model.fit(data, target, batch_size=3, epochs=10, validation_split=0.2)
