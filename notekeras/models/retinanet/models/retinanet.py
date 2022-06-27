import numpy as np
from notekeras.initializers import PriorProbability
from notekeras.layers.retinanet import (ClipBoxes, FilterDetections,
                                        RegressBoxes, UpSampleLike)
from notekeras.utils.image import read_image_batch
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, Add, Concatenate, Conv2D,
                                     Input, Permute, Reshape)
from tensorflow.keras.models import Model

from .. import layers
from ..utils.anchors import AnchorParameters, anchor_parameter_default
from ..utils.image import preprocess_image, resize_image


class RetinaNetModel(Model):
    def __init__(self, inputs, backbone_layers, num_classes, num_anchors=None, submodels=None, name='retinanet'):
        """ Construct a RetinaNet model on top of a backbone.

        This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

        Args
            inputs                  : keras.layers.Input (or list of) for the input to the model.
            num_classes             : Number of classes to classify.
            num_anchors             : Number of base anchors.
            create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
            submodels               : Submodels to run on each feature map (default is regression and classification submodels).
            name                    : Name of the model.

        Returns
            A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

            The order of the outputs is as defined in submodels:
            ```
            [
                regression, classification, other[0], other[1], ...
            ]
            ```
        """

        super(RetinaNetModel, self).__init__()
        if num_anchors is None:
            num_anchors = AnchorParameters.default.num_anchors()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        if submodels is None:
            submodels = self.default_submodels()

        C3, C4, C5 = backbone_layers

        # compute pyramid features as per https://arxiv.org/abs/1708.02002
        features = self.create_pyramid_features(C3, C4, C5)

        # for all pyramid levels, run available submodels
        pyramids = self.build_pyramid(submodels, features)

        super(RetinaNetModel, self).__init__(inputs=inputs, outputs=pyramids, name=name)

    def build_pyramid(self, models, features):
        """ Applies all submodels to each FPN level.

        Args
            models   : List of submodels to run on each pyramid level (by default only regression, classifcation).
            features : The FPN features.

        Returns
            A list of tensors, one for each submodel.
        """
        return [self.build_model_pyramid(n, m, features) for n, m in models]

    def build_model_pyramid(self, name, model, features):
        """ Applies a single submodel to each FPN level.
        Args
            name     : Name of the submodel.
            model    : The submodel to evaluate.
            features : The FPN features.

        Returns
            A tensor containing the response from the submodel on the FPN features.
        """
        return Concatenate(axis=1, name=name)([model(f) for f in features])

    def create_pyramid_features(self, C3, C4, C5, feature_size=256):
        """ Creates the FPN layers on top of the backbone features.

        Args
            C3           : Feature stage C3 from the backbone.
            C4           : Feature stage C4 from the backbone.
            C5           : Feature stage C5 from the backbone.
            feature_size : The feature size to use for the resulting feature levels.

        Returns
            A list of feature levels [P3, P4, P5, P6, P7].
        """
        # upsample C5 to get P5 from the FPN paper
        P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
        P5_upsampled = UpSampleLike(name='P5_upsampled')([P5, C4])
        P5 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

        # add P5 elementwise to C4
        P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
        P4 = Add(name='P4_merged')([P5_upsampled, P4])
        P4_upsampled = UpSampleLike(name='P4_upsampled')([P4, C3])
        P4 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

        # add P4 elementwise to C3
        P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
        P3 = Add(name='P3_merged')([P4_upsampled, P3])
        P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        P6 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        P7 = Activation('relu', name='C6_relu')(P6)
        P7 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

        return [P3, P4, P5, P6, P7]

    def default_submodels(self):
        """ Create a list of default submodels used for object detection.

        The default submodels contains a regression submodel and a classification submodel.

        Args
            num_classes : Number of classes to use.
            num_anchors : Number of base anchors.

        Returns
            A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
        """
        return [('regression', self.default_regression_model(4)),
                ('classification', self.default_classification_model())]

    def default_regression_model(self, num_values, pyramid_feature_size=256, regression_feature_size=256,
                                 name='regression_submodel'):
        """ Creates the default regression submodel.

        Args
            num_values              : Number of values to regress.
            num_anchors             : Number of anchors to regress for each feature level.
            pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
            regression_feature_size : The number of filters to use in the layers in the regression submodel.
            name                    : The name of the submodel.

        Returns
            A keras.models.Model that predicts regression values for each anchor.
        """
        # All new conv layers except the final one in the
        # RetinaNet (classification) subnets are initialized
        # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'kernel_initializer': keras.initializers.glorot_normal(seed=None),
            'bias_initializer': 'zeros'
        }

        if K.image_data_format() == 'channels_first':
            inputs = Input(shape=(pyramid_feature_size, None, None))
        else:
            inputs = Input(shape=(None, None, pyramid_feature_size))
        outputs = inputs
        for i in range(4):
            outputs = Conv2D(
                filters=regression_feature_size,
                activation='relu',
                name='pyramid_regression_{}'.format(i),
                **options
            )(outputs)

        outputs = Conv2D(self.num_anchors * num_values, name='pyramid_regression', **options)(outputs)
        if K.image_data_format() == 'channels_first':
            outputs = Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
        outputs = Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

        return Model(inputs=inputs, outputs=outputs, name=name)

    def default_classification_model(self, pyramid_feature_size=256, prior_probability=0.01,
                                     classification_feature_size=256, name='classification_submodel'):
        """ Creates the default classification submodel.

        Args
            num_classes                 : Number of classes to predict a score for at each feature level.
            num_anchors                 : Number of anchors to predict classification scores for at each feature level.
            pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
            classification_feature_size : The number of filters to use in the layers in the classification submodel.
            name                        : The name of the submodel.

        Returns
            A keras.models.Model that predicts classes for each anchor.
        """
        options = {'kernel_size': 3, 'strides': 1, 'padding': 'same', }

        if K.image_data_format() == 'channels_first':
            inputs = Input(shape=(pyramid_feature_size, None, None))
        else:
            inputs = Input(shape=(None, None, pyramid_feature_size))
        outputs = inputs
        for i in range(4):
            outputs = Conv2D(filters=classification_feature_size,
                             activation='relu',
                             name='pyramid_classification_{}'.format(i),
                             kernel_initializer=keras.initializers.glorot_normal(seed=None),
                             bias_initializer='zeros',
                             **options
                             )(outputs)

        outputs = Conv2D(filters=self.num_classes * self.num_anchors,
                         kernel_initializer=keras.initializers.glorot_normal(seed=None),
                         bias_initializer=PriorProbability(probability=prior_probability),
                         name='pyramid_classification',
                         **options
                         )(outputs)

        # reshape output and apply sigmoid
        if K.image_data_format() == 'channels_first':
            outputs = Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
        outputs = Reshape((-1, self.num_classes), name='pyramid_classification_reshape')(outputs)
        outputs = Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

        return Model(inputs=inputs, outputs=outputs, name=name)


class RetinaNetBox(Model):
    def __init__(self, model=None, nms=True, class_specific_filter=True, name='retinanet-bbox', anchor_params=None,
                 nms_threshold=0.5, score_threshold=0.05, max_detections=300, parallel_iterations=32, **kwargs):
        """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

        This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
        These layers include applying the regression values to the anchors and performing NMS.

        Args
            model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
            nms                   : Whether to use non-maximum suppression for the filtering step.
            class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
            name                  : Name of the model.
            anchor_params         : Struct containing anchor parameters. If None, default values are used.
            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            max_detections        : Maximum number of detections to keep.
            parallel_iterations   : Number of batch items to process in parallel.
            **kwargs              : Additional kwargs to pass to the minimal retinanet model.

        Returns
            A keras.models.Model which takes an image as input and outputs the detections on the image.

            The order is defined as follows:
            ```
            [
                boxes, scores, labels, other[0], other[1], ...
            ]
            ```
        """
        super(RetinaNetBox, self).__init__(name=name)

        anchor_params = anchor_params or anchor_parameter_default

        # create RetinaNet model
        if model is None:
            model = RetinaNetModel(num_anchors=anchor_params.num_anchors(), **kwargs)

        # compute the anchors
        features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
        anchors = self.build_anchors(anchor_params, features)

        # we expect the anchors, regression and classification values as first output
        regression = model.outputs[0]
        classification = model.outputs[1]

        # "other" can be any additional output from custom submodels, by default this will be []
        other = model.outputs[2:]

        # apply predicted regression to anchors
        boxes = RegressBoxes(name='boxes')([anchors, regression])
        boxes = ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

        # filter detections (apply NMS / score threshold / select top-k)
        detections = FilterDetections(nms=nms,
                                      class_specific_filter=class_specific_filter,
                                      name='filtered_detections',
                                      nms_threshold=nms_threshold,
                                      score_threshold=score_threshold,
                                      max_detections=max_detections,
                                      parallel_iterations=parallel_iterations
                                      )([boxes, classification] + other)

        # construct the model
        super(RetinaNetBox, self).__init__(inputs=model.inputs, outputs=detections, name=name)

    def build_anchors(self, anchor_parameters, features):
        """ Builds anchors for the shape of the features from FPN.

        Args
            anchor_parameters : Parameteres that determine how anchors are generated.
            features          : The FPN features.

        Returns
            A tensor containing the anchors for the FPN features.

            The shape is:
            ```
            (batch_size, num_anchors, 4)
            ```
        """
        anchors = [
            layers.Anchors(size=anchor_parameters.sizes[i],
                           stride=anchor_parameters.strides[i],
                           ratios=anchor_parameters.ratios,
                           scales=anchor_parameters.scales,
                           name='anchors_{}'.format(i)
                           )(f) for i, f in enumerate(features)
        ]

        return Concatenate(axis=1, name='anchors')(anchors)

    def predict_result_batch(self, image_paths):
        images = read_image_batch(image_paths)

        scales = []
        image_inputs = []
        for image in images:
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocess_image(image)
            image, scale = resize_image(image)

            scales.append(scale)
            image_inputs.append(image)

        boxes, scores, labels = self.predict(np.array(image_inputs))

        for i in range(0, len(boxes)):
            boxes[i] = boxes[i] / scales[i]

        return boxes, scores, labels
