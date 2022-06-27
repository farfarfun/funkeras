import tensorflow as tf

from notekeras.component import Component


class SPPNet(Component):
    def __init__(self, pool_size, strides=1, padding='SAME', data_format=None, *args, **kwargs):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format

        self.pool_list = None
        super(SPPNet, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.pool_list = []

    def call(self, inputs, **kwargs):
        pool_list = []
        for ksize in self.pool_size:
            if ksize == 1:
                pool_list.append(inputs)
            else:
                pool_list.append(tf.nn.max_pool(inputs, ksize=ksize, padding=self.padding, strides=self.strides))

        output = tf.concat(pool_list, axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[-1] = output_shape[-1] * len(self.pool_size)
        return output_shape
