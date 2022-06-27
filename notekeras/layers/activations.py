import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Layer


class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = None
        self.alpha = None

    def build(self, input_shape):
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, inputs, **kwargs):
        inputs_normed = self.bn(inputs)
        inputs_p = tf.sigmoid(inputs_normed)

        return self.alpha * (1.0 - inputs_p) * inputs + inputs_p * inputs
