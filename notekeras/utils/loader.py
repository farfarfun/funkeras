import h5py
from tensorflow.keras.models import Model
from tensorflow.python.keras.saving import hdf5_format

from notekeras.component import Component


def load_weights(model: Model, filepath, by_name=False, skip_mismatch=False):
    layers = []
    for layer in model.layers:
        if isinstance(layer, Component):
            layers.extend(layer.layers)
        else:
            layers.append(layer)

    with h5py.File(filepath, 'r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        hdf5_format.load_weights_from_hdf5_group(f, layers)
        hdf5_format.load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=skip_mismatch)
