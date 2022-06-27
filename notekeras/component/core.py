import numpy as np
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.models import Model


class ComponentLayer(Layer):
    def __init__(self, inputs=None, layer_depth=1, *args, **kwargs):
        """
        :param layer_depth:
        :param as_model:组件是否当做一个模型返回，当模型的时候，输入inputs必须非空
        :param inputs:模型输入，当as_model为True时传入
        :param args:
        :param kwargs:
        """
        super(ComponentLayer, self).__init__(*args, **kwargs)

        self.layer_depth = layer_depth
        self.input_layer = inputs

        super(ComponentLayer, self).__init__(*args, **kwargs)

    def __call__(self, inputs, **kwargs):
        if self.layer_depth <= 0:
            return super(ComponentLayer, self).__call__(inputs=inputs, **kwargs)
        else:
            self.build(np.shape(inputs)[1:])
            return self.call(inputs, **kwargs)


class Component(Model):
    def __init__(self, as_model=False, inputs=None, layer_depth=1, *args, **kwargs):
        """
        :param layer_depth:
        :param as_model:组件是否当做一个模型返回，当模型的时候，输入inputs必须非空
        :param inputs:模型输入，当as_model为True时传入
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.as_model = as_model

        self.layer_depth = layer_depth
        self.input_layer = inputs

        if self.as_model:
            if inputs is None:
                raise BaseException("when as_model is True, inputs must be not None")
            self.build(np.shape(inputs)[1:])
            self.outputs = self.call(inputs)
            self.output_layer = self.outputs
            super().__init__(inputs=inputs, outputs=self.outputs, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def __call__(self, inputs, **kwargs):
        if self.as_model:
            return super().__call__(inputs, **kwargs)

        if self.layer_depth <= 0:
            return super().__call__(inputs=inputs, **kwargs)
        else:
            self.build(np.shape(inputs)[1:])
            return self.call(inputs, **kwargs)


class AutoEncoder(Component):
    def __init__(self, name='AutoEncoder', encode_size=None, *args, **kwargs):
        self.encode_size = encode_size or [128, 64, 32]
        super(AutoEncoder, self).__init__(name=name, *args, **kwargs)

    def _build(self, inputs):
        output = inputs
        for size in self.encode_size:
            output = Dense(size, activation='tanh')(output)
        return output

