import os
from distutils.util import strtobool

__all__ = [
    'keras', 'utils', 'activations', 'applications', 'backend', 'datasets', 'engine',
    'layers', 'preprocessing', 'wrappers', 'callbacks', 'constraints', 'initializers',
    'metrics', 'models', 'losses', 'optimizers', 'regularizers', 'TF_KERAS',
    'Layer', 'Dense'
]

TF_KERAS = strtobool(os.environ.get('TF_KERAS', '0'))

if TF_KERAS:
    from tensorflow.python import keras

    print("import keras from tensorflow")
else:
    import keras

    print("import keras from keras")

utils = keras.utils
activations = keras.activations
applications = keras.applications
backend = keras.backend
datasets = keras.datasets
engine = keras.engine
layers = keras.layers
preprocessing = keras.preprocessing
wrappers = keras.wrappers
callbacks = keras.callbacks
constraints = keras.constraints
initializers = keras.initializers
metrics = keras.metrics
models = keras.models
losses = keras.losses
optimizers = keras.optimizers
regularizers = keras.regularizers
Layer = keras.layers.Layer
Dense = keras.layers.Dense
