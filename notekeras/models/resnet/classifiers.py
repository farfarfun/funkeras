from tensorflow import keras
from tensorflow.keras.models import Model

from notekeras.models.resnet import core as models


class ResNet18Classifier(Model):
    """
    A :class:`ResNet18 <ResNet18>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:
        >>> from notekeras.models.resnet.classifiers import ResNet18Classifier
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = ResNet18Classifier(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """

    def __init__(self, inputs, classes):
        outputs = models.ResNet2D18(inputs)
        outputs = keras.layers.Flatten()(outputs.output)
        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
        super(ResNet18Classifier, self).__init__(inputs, outputs)


class ResNet34Classifier(Model):
    """
    A :class:`ResNet34 <ResNet34>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:
        >>> from notekeras.models.resnet.classifiers import ResNet34Classifier
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = ResNet34Classifier(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """

    def __init__(self, inputs, classes):
        outputs = models.ResNet2D34(inputs)

        outputs = keras.layers.Flatten()(outputs.output)

        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)

        super(ResNet34Classifier, self).__init__(inputs, outputs)


class ResNet50Classifier(Model):
    """
    A :class:`ResNet50 <ResNet50>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> from notekeras.models.resnet.classifiers import ResNet50Classifier
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = ResNet50Classifier(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """

    def __init__(self, inputs, classes):
        outputs = models.ResNet2D50(inputs)
        outputs = keras.layers.Flatten()(outputs.output)
        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
        super(ResNet50Classifier, self).__init__(inputs, outputs)


class ResNet101Classifier(Model):
    """
    A :class:`ResNet101 <ResNet101>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:
        >>> from notekeras.models.resnet.classifiers import ResNet101Classifier
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = ResNet101Classifier(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """

    def __init__(self, inputs, classes):
        outputs = models.ResNet2D101(inputs)
        outputs = keras.layers.Flatten()(outputs.output)
        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
        super(ResNet101Classifier, self).__init__(inputs, outputs)


class ResNet152Classifier(Model):
    """
    A :class:`ResNet152 <ResNet152>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:
        >>> from notekeras.models.resnet.classifiers import ResNet152Classifier
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = ResNet152Classifier(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])

    """

    def __init__(self, inputs, classes):
        outputs = models.ResNet2D152(inputs)
        outputs = keras.layers.Flatten()(outputs.output)
        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
        super(ResNet152Classifier, self).__init__(inputs, outputs)


class ResNet200Classifier(Model):
    """
    A :class:`ResNet200 <ResNet200>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:
        >>> from notekeras.models.resnet.classifiers import ResNet200Classifier
        >>> shape, classes = (224, 224, 3), 1000
        >>> x = keras.layers.Input(shape)
        >>> model = ResNet200Classifier(x)
        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """

    def __init__(self, inputs, classes):
        outputs = models.ResNet2D200(inputs)
        outputs = keras.layers.Flatten()(outputs.output)
        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)
        super(ResNet200Classifier, self).__init__(inputs, outputs)
