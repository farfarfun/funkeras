from notekeras.backend import keras


class WideDeepComponent(keras.layers.Layer):
    def __init__(self,
                 name,
                 deep_shape=None,
                 deep_activation='relu'
                 ):
        super(WideDeepComponent, self).__init__()

        if deep_shape is None:
            deep_shape = [64, 64, 32]

        self.name = name
        self.deep_shape = deep_shape
        self.deep_activation = deep_activation

        self.attention_layer2 = None
        self.feed_forward_layer2 = None

        self.supports_masking = True

        self.deep_dense = []
        self._build()

    def _build(self):
        for shape in self.deep_shape:
            dense = keras.layers.Dense(shape, activation=self.deep_activation)
            self.deep_dense.append(dense)
        self.liner = keras.layers.Dense(1, activation='sigmoid')

    def __call__(self, inputs, **kwargs):
        input_deep, input_wide = inputs
        deep = input_deep

        for dense in self.deep_dense:
            deep = dense(deep)
        concat = keras.backend.concatenate(deep, input_wide)
        out = self.liner(concat)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'deep_shape': self.deep_shape,
            'deep_activation': self.deep_activation,
        }
        return config
