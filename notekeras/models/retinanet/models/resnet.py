import tensorflow.keras.backend as K
from notekeras.layers import BatchNormalizationFreeze
from notekeras.models.resnet import ResNet50, ResNet101, ResNet152
from notekeras.models.retinanet.models import Backbone, BackboneModel
from notekeras.models.retinanet.models.retinanet import RetinaNetModel
from notekeras.models.retinanet.utils.image import preprocess_image
from tensorflow import keras
from tensorflow.keras.utils import get_file


class ResNetBackboneModel(BackboneModel):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(ResNetBackboneModel, self).__init__(backbone)
        custom_objects = {
            'BatchNormalization': BatchNormalizationFreeze,
        }

        self.custom_objects.update(custom_objects)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        resnet_filename = 'ResNet-{}-model.keras.h5'
        resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
        depth = int(self.backbone.replace('resnet', ''))

        filename = resnet_filename.format(depth)
        resource = resnet_resource.format(depth)
        if depth == 50:
            checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
        elif depth == 101:
            checksum = '05dc86924389e5b401a9ea0348a3213c'
        elif depth == 152:
            checksum = '6ee11ef2b135592f8031058820bb9e71'

        return get_file(filename, resource, cache_subdir='models', md5_hash=checksum)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnet50', 'resnet101', 'resnet152']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


class ResNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__(backbone)
        custom_objects = {
            'BatchNormalization': BatchNormalizationFreeze,
        }

        self.custom_objects.update(custom_objects)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        resnet_filename = 'ResNet-{}-model.keras.h5'
        resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
        depth = int(self.backbone.replace('resnet', ''))

        filename = resnet_filename.format(depth)
        resource = resnet_resource.format(depth)
        if depth == 50:
            checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
        elif depth == 101:
            checksum = '05dc86924389e5b401a9ea0348a3213c'
        elif depth == 152:
            checksum = '6ee11ef2b135592f8031058820bb9e71'

        return get_file(
            filename,
            resource,
            cache_subdir='models',
            md5_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnet50', 'resnet101', 'resnet152']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def resnet_retinanet(num_classes, backbone='resnet50', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a resnet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        if K.image_data_format() == 'channels_first':
            inputs = keras.layers.Input(shape=(3, None, None))
        else:
            inputs = keras.layers.Input(shape=(None, None, 3))

    # create the resnet backbone
    if backbone == 'resnet50':
        resnet = ResNet50(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet101':
        resnet = ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet152':
        resnet = ResNet152(inputs, include_top=False, freeze_bn=True)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    return RetinaNetModel(inputs=inputs, num_classes=num_classes, backbone_layers=resnet.outputs[1:], **kwargs)


def resnet50_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)


def resnet101_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet101', inputs=inputs, **kwargs)


def resnet152_retinanet(num_classes, inputs=None, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone='resnet152', inputs=inputs, **kwargs)
