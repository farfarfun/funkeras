import tensorflow as tf
from notekeras.backend import backend as K
from notekeras.backend import layers


class Masked(layers.Layer):
    """Generate output mask based on the given mask.

    The inputs for the layer is the original input layer and the masked locations.

    See: https://arxiv.org/pdf/1810.04805.pdf
    """

    def __init__(self,
                 return_masked=False,
                 **kwargs):
        """Initialize the layer.

        :param return_masked: Whether to return the merged mask.
        :param kwargs: Arguments for parent class.
        """
        super(Masked, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_masked = return_masked

    def get_config(self):
        config = {
            'return_masked': self.return_masked,
        }
        base_config = super(Masked, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.return_masked:
            return [input_shape[0], input_shape[0][:-1]]
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        token_mask = K.not_equal(inputs[1], 0)
        if mask is None:
            masked = token_mask
        else:
            masked = K.all(K.stack([token_mask, mask[0]], axis=0), axis=0)
        if self.return_masked:
            return [masked, None]
        return masked

    def call(self, inputs, mask=None, **kwargs):
        output = tf.identity(inputs[0])
        if self.return_masked:
            return [output, K.cast(self.compute_mask(inputs, mask)[0], K.floatx())]
        return output
