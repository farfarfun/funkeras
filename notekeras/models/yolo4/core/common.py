import tensorflow as tf

from notekeras.component.yolo.core import YoloConv, YoloResBlock


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    conv = YoloConv(filters=filters_shape[-1], kernel_size=filters_shape[0], down_sample=downsample, activate=activate,
                    bn=bn, activate_type=activate_type)
    return conv(input_layer)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)


def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    return YoloResBlock(input_channel=input_channel, filter_num1=filter_num1, filter_num2=filter_num2,
                        activate_type=activate_type)(input_layer)


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')
