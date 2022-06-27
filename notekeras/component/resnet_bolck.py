from notekeras.layers import BatchNormalizationFreeze
from tensorflow import keras
from tensorflow.keras import layers

parameters = {
    "kernel_initializer": "he_normal"
}


def basic_1d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):
    """
    A one-dimensional basic block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.basic_1d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = -1
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = layers.ZeroPadding1D(
            padding=1,
            name="padding{}{}_branch2a".format(stage_char, block_char)
        )(x)

        y = layers.Conv1D(
            filters,
            kernel_size,
            strides=stride,
            use_bias=False,
            name="res{}{}_branch2a".format(stage_char, block_char),
            **parameters
        )(y)

        y = BatchNormalizationFreeze(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2a".format(stage_char, block_char)
        )(y)

        y = layers.Activation(
            "relu",
            name="res{}{}_branch2a_relu".format(stage_char, block_char)
        )(y)

        y = layers.ZeroPadding1D(
            padding=1,
            name="padding{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = layers.Conv1D(
            filters,
            kernel_size,
            use_bias=False,
            name="res{}{}_branch2b".format(stage_char, block_char),
            **parameters
        )(y)

        y = BatchNormalizationFreeze(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2b".format(stage_char, block_char)
        )(y)

        if block == 0:
            shortcut = layers.Conv1D(
                filters,
                1,
                strides=stride,
                use_bias=False,
                name="res{}{}_branch1".format(stage_char, block_char),
                **parameters
            )(x)

            shortcut = BatchNormalizationFreeze(
                axis=axis,
                epsilon=1e-5,
                freeze=freeze_bn,
                name="bn{}{}_branch1".format(stage_char, block_char)
            )(shortcut)
        else:
            shortcut = x

        y = layers.Add(
            name="res{}{}".format(stage_char, block_char)
        )([y, shortcut])

        y = layers.Activation(
            "relu",
            name="res{}{}_relu".format(stage_char, block_char)
        )(y)

        return y

    return f


def bottleneck_1d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):
    """
    A one-dimensional bottleneck block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.bottleneck_1d(64)
    """
    if stride is None:
        stride = 1 if block != 0 or stage == 0 else 2

    if keras.backend.image_data_format() == "channels_last":
        axis = -1
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = layers.Conv1D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            name="res{}{}_branch2a".format(stage_char, block_char),
            **parameters
        )(x)

        y = BatchNormalizationFreeze(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2a".format(stage_char, block_char)
        )(y)

        y = layers.Activation(
            "relu",
            name="res{}{}_branch2a_relu".format(stage_char, block_char)
        )(y)

        y = layers.ZeroPadding1D(
            padding=1,
            name="padding{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = layers.Conv1D(
            filters,
            kernel_size,
            use_bias=False,
            name="res{}{}_branch2b".format(stage_char, block_char),
            **parameters
        )(y)

        y = BatchNormalizationFreeze(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2b".format(stage_char, block_char)
        )(y)

        y = layers.Activation(
            "relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = layers.Conv1D(filters * 4, 1, use_bias=False,
                          name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalizationFreeze(
            axis=axis,
            epsilon=1e-5,
            freeze=freeze_bn,
            name="bn{}{}_branch2c".format(stage_char, block_char)
        )(y)

        if block == 0:
            shortcut = layers.Conv1D(
                filters * 4,
                1,
                strides=stride,
                use_bias=False,
                name="res{}{}_branch1".format(stage_char, block_char),
                **parameters
            )(x)

            shortcut = BatchNormalizationFreeze(
                axis=axis,
                epsilon=1e-5,
                freeze=freeze_bn,
                name="bn{}{}_branch1".format(stage_char, block_char)
            )(shortcut)
        else:
            shortcut = x

        y = layers.Add(
            name="res{}{}".format(stage_char, block_char)
        )([y, shortcut])

        y = layers.Activation(
            "relu",
            name="res{}{}_relu".format(stage_char, block_char)
        )(y)

        return y

    return f


def basic_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):
    """
    A two-dimensional basic block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.basic_2d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = layers.ZeroPadding2D(
            padding=1, name="padding{}{}_branch2a".format(stage_char, block_char))(x)

        y = layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False,
                          name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                     name="bn{}{}_branch2a".format(stage_char, block_char))(y)

        y = layers.Activation(
            "relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = layers.ZeroPadding2D(
            padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)

        y = layers.Conv2D(filters, kernel_size, use_bias=False,
                          name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                     name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False,
                                     name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)

            shortcut = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                                name="bn{}{}_branch1".format(stage_char, block_char))(
                shortcut)
        else:
            shortcut = x

        y = layers.Add(name="res{}{}".format(
            stage_char, block_char))([y, shortcut])

        y = layers.Activation(
            "relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def bottleneck_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):
    """
    A two-dimensional bottleneck block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.bottleneck_2d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False,
                          name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(x)

        y = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                     name="bn{}{}_branch2a".format(stage_char, block_char))(y)

        y = layers.Activation(
            "relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = layers.ZeroPadding2D(
            padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)

        y = layers.Conv2D(filters, kernel_size, use_bias=False,
                          name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                     name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        y = layers.Activation(
            "relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = layers.Conv2D(filters * 4, (1, 1), use_bias=False,
                          name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                     name="bn{}{}_branch2c".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = layers.Conv2D(filters * 4, (1, 1), strides=stride, use_bias=False,
                                     name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)

            shortcut = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                                name="bn{}{}_branch1".format(stage_char, block_char))(
                shortcut)
        else:
            shortcut = x

        y = layers.Add(name="res{}{}".format(
            stage_char, block_char))([y, shortcut])

        y = layers.Activation(
            "relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def basic_3d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):
    """
    A three-dimensional basic block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.basic_3d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = layers.ZeroPadding3D(
            padding=1, name="padding{}{}_branch2a".format(stage_char, block_char))(x)

        y = layers.Conv3D(filters, kernel_size, strides=stride, use_bias=False,
                          name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                     name="bn{}{}_branch2a".format(stage_char, block_char))(y)

        y = layers.Activation(
            "relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = layers.ZeroPadding3D(
            padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)

        y = layers.Conv3D(filters, kernel_size, use_bias=False,
                          name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                     name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = layers.Conv3D(filters, (1, 1), strides=stride, use_bias=False,
                                     name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)

            shortcut = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                                name="bn{}{}_branch1".format(stage_char, block_char))(
                shortcut)
        else:
            shortcut = x

        y = layers.Add(name="res{}{}".format(
            stage_char, block_char))([y, shortcut])

        y = layers.Activation(
            "relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def bottleneck_3d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None, freeze_bn=False):
    """
    A three-dimensional bottleneck block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.bottleneck_3d(64)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = layers.Conv3D(filters, (1, 1), strides=stride, use_bias=False,
                          name="res{}{}_branch2a".format(stage_char, block_char), **parameters)(x)

        y = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                     name="bn{}{}_branch2a".format(stage_char, block_char))(y)

        y = layers.Activation(
            "relu", name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = layers.ZeroPadding3D(
            padding=1, name="padding{}{}_branch2b".format(stage_char, block_char))(y)

        y = layers.Conv3D(filters, kernel_size, use_bias=False,
                          name="res{}{}_branch2b".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                     name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        y = layers.Activation(
            "relu", name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = layers.Conv3D(filters * 4, (1, 1), use_bias=False,
                          name="res{}{}_branch2c".format(stage_char, block_char), **parameters)(y)

        y = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                     name="bn{}{}_branch2c".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = layers.Conv3D(filters * 4, (1, 1), strides=stride, use_bias=False,
                                     name="res{}{}_branch1".format(stage_char, block_char), **parameters)(x)

            shortcut = BatchNormalizationFreeze(axis=axis, epsilon=1e-5, freeze=freeze_bn,
                                                name="bn{}{}_branch1".format(stage_char, block_char))(
                shortcut)
        else:
            shortcut = x

        y = layers.Add(name="res{}{}".format(
            stage_char, block_char))([y, shortcut])

        y = layers.Activation(
            "relu", name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def time_distributed_basic_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None,
                              freeze_bn=False):
    """

    A time distributed two-dimensional basic block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.time_distributed_basic_2d(64)

    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = layers.TimeDistributed(layers.ZeroPadding2D(padding=1),
                                   name="padding{}{}_branch2a".format(stage_char, block_char))(x)

        y = layers.TimeDistributed(
            layers.Conv2D(filters, kernel_size,
                          strides=stride, use_bias=False, **parameters),
            name="res{}{}_branch2a".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(
            BatchNormalizationFreeze(
                axis=axis, epsilon=1e-5, freeze=freeze_bn),
            name="bn{}{}_branch2a".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(layers.Activation("relu"),
                                   name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(layers.ZeroPadding2D(padding=1),
                                   name="padding{}{}_branch2b".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(layers.Conv2D(filters, kernel_size, use_bias=False, **parameters),
                                   name="res{}{}_branch2b".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(
            BatchNormalizationFreeze(
                axis=axis, epsilon=1e-5, freeze=freeze_bn),
            name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = layers.TimeDistributed(
                layers.Conv2D(
                    filters, (1, 1), strides=stride, use_bias=False, **parameters),
                name="res{}{}_branch1".format(stage_char, block_char))(x)

            shortcut = layers.TimeDistributed(
                BatchNormalizationFreeze(
                    axis=axis, epsilon=1e-5, freeze=freeze_bn),
                name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = layers.Add(name="res{}{}".format(
            stage_char, block_char))([y, shortcut])

        y = layers.TimeDistributed(layers.Activation("relu"),
                                   name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f


def time_distributed_bottleneck_2d(filters, stage=0, block=0, kernel_size=3, numerical_name=False, stride=None,
                                   freeze_bn=False):
    """

    A time distributed two-dimensional bottleneck block.

    :param filters: the output’s feature space

    :param stage: int representing the stage of this block (starting from 0)

    :param block: int representing this block (starting from 0)

    :param kernel_size: size of the kernel

    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})

    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id

    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)

    Usage:

        >>> import keras_resnet.blocks

        >>> keras_resnet.blocks.time_distributed_bottleneck_2d(64)

    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if keras.backend.image_data_format() == "channels_last":
        axis = 3
    else:
        axis = 1

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def f(x):
        y = layers.TimeDistributed(
            layers.Conv2D(filters, (1, 1), strides=stride,
                          use_bias=False, **parameters),
            name="res{}{}_branch2a".format(stage_char, block_char))(x)

        y = layers.TimeDistributed(
            BatchNormalizationFreeze(
                axis=axis, epsilon=1e-5, freeze=freeze_bn),
            name="bn{}{}_branch2a".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(layers.Activation("relu"),
                                   name="res{}{}_branch2a_relu".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(layers.ZeroPadding2D(padding=1),
                                   name="padding{}{}_branch2b".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(layers.Conv2D(filters, kernel_size, use_bias=False, **parameters),
                                   name="res{}{}_branch2b".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(
            BatchNormalizationFreeze(
                axis=axis, epsilon=1e-5, freeze=freeze_bn),
            name="bn{}{}_branch2b".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(layers.Activation("relu"),
                                   name="res{}{}_branch2b_relu".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(layers.Conv2D(filters * 4, (1, 1), use_bias=False, **parameters),
                                   name="res{}{}_branch2c".format(stage_char, block_char))(y)

        y = layers.TimeDistributed(
            BatchNormalizationFreeze(
                axis=axis, epsilon=1e-5, freeze=freeze_bn),
            name="bn{}{}_branch2c".format(stage_char, block_char))(y)

        if block == 0:
            shortcut = layers.TimeDistributed(
                layers.Conv2D(
                    filters * 4, (1, 1), strides=stride, use_bias=False, **parameters),
                name="res{}{}_branch1".format(stage_char, block_char))(x)

            shortcut = layers.TimeDistributed(
                layers.BatchNormalization(
                    axis=axis, epsilon=1e-5, freeze=freeze_bn),
                name="bn{}{}_branch1".format(stage_char, block_char))(shortcut)
        else:
            shortcut = x

        y = layers.Add(name="res{}{}".format(
            stage_char, block_char))([y, shortcut])

        y = layers.TimeDistributed(layers.Activation("relu"),
                                   name="res{}{}_relu".format(stage_char, block_char))(y)

        return y

    return f
