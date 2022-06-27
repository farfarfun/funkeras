import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D

from notekeras.component import Component
from notekeras.utils import compose


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def up_sample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)


class YoloConv(Component):
    def __init__(self,
                 filters,
                 kernel_size,
                 down_sample=False,
                 activate=True,
                 bn=True,
                 activate_type='leaky',
                 *args, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size

        self.down_sample = down_sample
        self.activate = activate
        self.bn = bn
        self.activate_type = activate_type

        self.compose_list = None

        kwargs.update({"layer_depth": kwargs.get("layer_depth", 1)})
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.compose_list = []
        if self.down_sample:
            self.compose_list.append(ZeroPadding2D(((1, 0), (1, 0))))
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        self.compose_list.append(Conv2D(filters=self.filters,
                                        kernel_size=self.kernel_size,
                                        strides=strides,
                                        padding=padding,
                                        use_bias=not self.bn,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        bias_initializer=tf.constant_initializer(0.)))

        if self.bn:
            self.compose_list.append(BatchNormalization())

        self.layers.extend(self.compose_list)

    def call(self, inputs, **kwargs):
        output = compose(*self.compose_list)(inputs)

        if self.activate:
            if self.activate_type == "leaky":
                output = tf.nn.leaky_relu(output, alpha=0.1)
            elif self.activate_type == "mish":
                output = mish(output)

        return output

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)


class YoloResBlock(Component):
    def __init__(self, input_channel, filter_num1, filter_num2, name='YoloResidualBlock', activate_type='leaky', *args,
                 **kwargs):
        super(YoloResBlock, self).__init__(name=name)
        self.input_channel = input_channel
        self.filter_num1 = filter_num1
        self.filter_num2 = filter_num2
        self.activate_type = activate_type

        self.compose_list = []
        super(YoloResBlock, self).__init__(name=name, *args, **kwargs)

    def build(self, input_shape):
        self.compose_list = [YoloConv(filters=self.filter_num1, kernel_size=1, activate_type=self.activate_type),
                             YoloConv(filters=self.filter_num2, kernel_size=3, activate_type=self.activate_type)]

    def call(self, inputs, **kwargs):
        residual_output = inputs + compose(*self.compose_list)(inputs)
        return residual_output


def YoloCSPX(input_data, rec_num, filters=64, kernel_size=1, activate_type='mish'):
    route_data = YoloConv(filters=filters, kernel_size=kernel_size, activate_type=activate_type)(input_data)
    input_data = YoloConv(filters=filters, kernel_size=kernel_size, activate_type=activate_type)(input_data)

    for i in range(rec_num):
        input_data = YoloResBlock(input_channel=filters, filter_num1=filters, filter_num2=filters,
                                  activate_type=activate_type)(input_data)
    input_data = YoloConv(filters=filters, kernel_size=kernel_size, activate_type=activate_type)(input_data)

    return tf.concat([input_data, route_data], axis=-1)


def YoloNeck(conv, route_tmp, filters):
    conv = tf.concat([conv, route_tmp], axis=-1)

    conv = compose(YoloConv(filters=filters, kernel_size=1),
                   YoloConv(filters=filters * 2, kernel_size=3),
                   YoloConv(filters=filters, kernel_size=1),
                   YoloConv(filters=filters * 2, kernel_size=3),
                   YoloConv(filters=filters, kernel_size=1))(conv)

    return conv
