import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D,Conv2D

import notekeras.models.yolo4.core.common as common
from notekeras.component.image.core import SPPNet
from notekeras.component.yolo.core import YoloConv, YoloResBlock, YoloCSPX


def darknet53(input_data):
    input_data = YoloConv(filters=32, kernel_size=3, down_sample=True)(input_data)

    input_data = YoloConv(filters=64, kernel_size=3, down_sample=True)(input_data)
    for i in range(1):
        input_data = YoloResBlock(input_channel=64, filter_num1=32, filter_num2=64)(input_data)

    input_data = YoloConv(filters=128, kernel_size=3, down_sample=True)(input_data)
    for i in range(2):
        input_data = YoloResBlock(input_channel=128, filter_num1=64, filter_num2=128)(input_data)

    input_data = YoloConv(filters=256, kernel_size=3, down_sample=True)(input_data)
    for i in range(8):
        input_data = YoloResBlock(input_channel=256, filter_num1=128, filter_num2=256)(input_data)

    route_1 = input_data

    input_data = YoloConv(filters=512, kernel_size=3, down_sample=True)(input_data)
    for i in range(8):
        input_data = YoloResBlock(input_channel=512, filter_num1=256, filter_num2=512)(input_data)

    route_2 = input_data

    input_data = YoloConv(filters=1024, kernel_size=3, down_sample=True)(input_data)
    for i in range(4):
        input_data = YoloResBlock(input_channel=1024, filter_num1=512, filter_num2=1024)(input_data)

    return route_1, route_2, input_data


def cspdarknet53(input_data):
    input_data = YoloConv(filters=32, kernel_size=3, activate_type='mish')(input_data)
    input_data = YoloConv(filters=64, kernel_size=3, activate_type='mish', down_sample=True)(input_data)
    route_data = YoloConv(filters=64, kernel_size=1, activate_type='mish')(input_data)
    input_data = YoloConv(filters=64, kernel_size=1, activate_type='mish')(input_data)

    for i in range(1):
        input_data = YoloResBlock(input_channel=64, filter_num1=32, filter_num2=64, activate_type='mish')(input_data)
    input_data = YoloConv(filters=64, kernel_size=1, activate_type='mish')(input_data)
    input_data = tf.concat([input_data, route_data], axis=-1)

    #################
    input_data = YoloConv(filters=64, kernel_size=1, activate_type='mish')(input_data)
    input_data = YoloConv(filters=128, kernel_size=3, activate_type='mish', down_sample=True)(input_data)
    input_data = YoloCSPX(input_data, rec_num=2, filters=64, kernel_size=1, activate_type='mish')

    #################
    input_data = YoloConv(filters=128, kernel_size=1, activate_type='mish')(input_data)
    input_data = YoloConv(filters=256, kernel_size=3, activate_type='mish', down_sample=True)(input_data)
    input_data = YoloCSPX(input_data, rec_num=8, filters=128, kernel_size=1, activate_type='mish')

    #################
    input_data = YoloConv(filters=256, kernel_size=1, activate_type='mish')(input_data)
    route_1 = input_data
    input_data = YoloConv(filters=512, kernel_size=3, activate_type='mish', down_sample=True)(input_data)
    input_data = YoloCSPX(input_data, rec_num=8, filters=256, kernel_size=1, activate_type='mish')

    #################
    input_data = YoloConv(filters=512, kernel_size=1, activate_type='mish')(input_data)
    route_2 = input_data
    input_data = YoloConv(filters=1024, kernel_size=3, activate_type='mish', down_sample=True)(input_data)
    input_data = YoloCSPX(input_data, rec_num=4, filters=512, kernel_size=1, activate_type='mish')

    #################
    input_data = YoloConv(filters=1024, kernel_size=1, activate_type='mish')(input_data)

    input_data = YoloConv(filters=512, kernel_size=1)(input_data)
    input_data = YoloConv(filters=1024, kernel_size=3)(input_data)
    input_data = YoloConv(filters=512, kernel_size=1)(input_data)

    input_data = SPPNet(pool_size=[13, 9, 5, 1], padding='SAME', strides=1)(input_data)

    input_data = YoloConv(filters=512, kernel_size=1)(input_data)
    input_data = YoloConv(filters=1024, kernel_size=3)(input_data)
    route_3 = YoloConv(filters=512, kernel_size=1)(input_data)

    return route_1, route_2, route_3


def cspdarknet53_tiny(input_data):
    input_data = YoloConv(filters=32, kernel_size=3, down_sample=True)(input_data)
    input_data = YoloConv(filters=64, kernel_size=3, down_sample=True)(input_data)
    input_data = YoloConv(filters=64, kernel_size=3)(input_data)

    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = YoloConv(filters=32, kernel_size=3)(input_data)
    route_1 = input_data
    input_data = YoloConv(filters=32, kernel_size=3)(input_data)
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = YoloConv(filters=64, kernel_size=1)(input_data)
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = YoloConv(filters=128, kernel_size=3)(input_data)

    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = YoloConv(filters=64, kernel_size=3)(input_data)
    route_1 = input_data
    input_data = YoloConv(filters=64, kernel_size=3)(input_data)
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = YoloConv(filters=128, kernel_size=1)(input_data)
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = YoloConv(filters=256, kernel_size=3)(input_data)

    route = input_data
    input_data = common.route_group(input_data, 2, 1)
    input_data = YoloConv(filters=128, kernel_size=3)(input_data)
    route_1 = input_data
    input_data = YoloConv(filters=128, kernel_size=3)(input_data)
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = YoloConv(filters=256, kernel_size=1)(input_data)
    route_1 = input_data
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = YoloConv(filters=512, kernel_size=3)(input_data)

    return route_1, input_data


def darknet53_tiny(input_data):
    input_data = YoloConv(filters=16, kernel_size=3)(input_data)
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = YoloConv(filters=32, kernel_size=3)(input_data)
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = YoloConv(filters=64, kernel_size=3)(input_data)
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = YoloConv(filters=128, kernel_size=3)(input_data)
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = YoloConv(filters=256, kernel_size=3)(input_data)

    route_1 = input_data
    input_data = MaxPool2D(2, 2, 'same')(input_data)
    input_data = YoloConv(filters=512, kernel_size=3)(input_data)
    input_data = MaxPool2D(2, 1, 'same')(input_data)
    input_data = YoloConv(filters=1024, kernel_size=3)(input_data)

    return route_1, input_data
