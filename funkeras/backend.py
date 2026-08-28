import os

from tensorflow import keras
from tensorflow.keras.backend import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__all__ = [
    'keras', 'utils', 'activations', 'applications', 'backend', 'datasets',
    'layers', 'preprocessing', 'wrappers', 'callbacks', 'constraints', 'initializers',
    'metrics', 'models', 'losses', 'optimizers', 'regularizers',
    'Dense', 'plot_model',
    'Layer', 'Model',
]

# keras.backend.set_floatx('float64')
utils = keras.utils
activations = keras.activations
applications = keras.applications
backend = keras.backend
datasets = keras.datasets

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
plot_model = keras.utils.plot_model

layers = keras.layers
Layer = layers.Layer
Dense = layers.Dense
Model = keras.models.Model
