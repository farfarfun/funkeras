import sys

from notekeras.initializers import PriorProbability
from notekeras.layers.retinanet import (ClipBoxes, FilterDetections,
                                        RegressBoxes, UpSampleLike)
from notekeras.models.retinanet import layers
from notekeras.models.retinanet.losses import focal, smooth_l1
from notekeras.models.retinanet.models import RetinaNetBox


class Backbone(object):
    """ This class stores additional information on backbones.
    """

    def __init__(self, backbone):
        self.custom_objects = {
            'UpsampleLike': UpSampleLike,
            'PriorProbability': PriorProbability,
            'RegressBoxes': RegressBoxes,
            'FilterDetections': FilterDetections,
            'Anchors': layers.Anchors,
            'ClipBoxes': ClipBoxes,
            '_smooth_l1': smooth_l1(),
            '_focal': focal(),
        }

        self.backbone = backbone
        self.validate()

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('retinanet method not implemented.')

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        raise NotImplementedError('download_imagenet method not implemented.')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')


def backbone(backbone_name):
    """ Returns a backbone object for the given backbone.
    """
    if 'densenet' in backbone_name:
        from .densenet import DenseNetBackbone as b
    elif 'seresnext' in backbone_name or 'seresnet' in backbone_name or 'senet' in backbone_name:
        from .senet import SeBackbone as b
    elif 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    elif 'mobilenet' in backbone_name:
        from .mobilenet import MobileNetBackbone as b
    elif 'vgg' in backbone_name:
        from .vgg import VGGBackbone as b
    elif 'EfficientNet' in backbone_name:
        from .effnet import EfficientNetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name)


def load_model(filepath, backbone_name='resnet50'):
    """ Loads a retinanet model using the correct custom objects.

    Args
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone_name         : Backbone with which the model was trained.

    Returns
        A keras.models.Model object.

    Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    import tensorflow.keras.models
    return tensorflow.keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)


def convert_model(model, nms=True, class_specific_filter=True, anchor_params=None, **kwargs):
    """ Converts a training model to an inference model.

    Args
        model                 : A retinanet training model.
        nms                   : Boolean, whether to add NMS filtering to the converted model.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        anchor_params         : Anchor parameters object. If omitted, default values are used.
        **kwargs              : Inference and minimal retinanet model settings.

    Returns
        A keras.models.Model object.

    Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """

    return RetinaNetBox(model=model, nms=nms, class_specific_filter=class_specific_filter,
                        anchor_params=anchor_params, **kwargs)


def assert_training_model(model):
    """ Assert that the model is a training model.
    """
    assert (all(output in model.output_names for output in ['regression', 'classification'])), \
        "Input is not a training model (no 'regression' and 'classification' outputs were found, outputs are: {}).".format(
            model.output_names)


def check_training_model(model):
    """ Check that model is a training model and exit otherwise.
    """
    try:
        assert_training_model(model)
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
