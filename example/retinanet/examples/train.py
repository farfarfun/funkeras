import os

from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau

from notekeras.model.retinanet import models
from notekeras.model.retinanet.generator import TextGenerator
from notekeras.model.retinanet.losses import smooth_l1, focal
from notekeras.model.retinanet.models.retinanet import RetinaNetBox
from notekeras.model.retinanet.utils.image import random_visual_effect_generator
from notekeras.model.retinanet.utils.transform import random_transform_generator
from notekeras.utils.model import freeze


def create_models(backbone_retinanet, num_classes, weights=None, freeze_backbone=False, lr=1e-5):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze if freeze_backbone else None

    anchor_params = None
    num_anchors = None

    model = backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier)
    if weights:
        model.load_weights(weights=weights, by_name=True, skip_mismatch=True)

    training_model = model

    prediction_model = RetinaNetBox(model=model, anchor_params=anchor_params)

    training_model.compile(loss={'regression': smooth_l1(), 'classification': focal()},
                           optimizer=keras.optimizers.Adam(lr=lr, clipnorm=0.001))

    return model, training_model, prediction_model


def create_callbacks(batch_size, tensorboard_dir=None):
    callbacks = []

    callbacks.append(ReduceLROnPlateau(monitor='loss',
                                       factor=0.1,
                                       patience=2,
                                       verbose=1,
                                       mode='auto',
                                       min_delta=0.0001,
                                       cooldown=0,
                                       min_lr=0
                                       ))

    if tensorboard_dir:
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        tensorboard_callback = TensorBoard(log_dir=tensorboard_dir,
                                           histogram_freq=0,
                                           batch_size=batch_size,
                                           write_graph=True,
                                           write_grads=False,
                                           write_images=False,
                                           embeddings_freq=0,
                                           embeddings_layer_names=None,
                                           embeddings_metadata=None
                                           )
        callbacks.append(tensorboard_callback)

    return callbacks


def create_generators(preprocess_image, batch_size=32, annotations=None, classes=None, random_transform=None,
                      val_annotations=None, config=None):
    """ Create generators for training and validation.
    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
        annotations      : Path to CSV file containing annotations for training.
        classes          : Path to a CSV file containing class label mapping.
    """
    common_args = {'batch_size': batch_size, 'config': config, 'preprocess_image': preprocess_image, }

    if random_transform:
        transform_generator = random_transform_generator(min_rotation=-0.1,
                                                         max_rotation=0.1,
                                                         min_translation=(-0.1, -0.1),
                                                         max_translation=(0.1, 0.1),
                                                         min_shear=-0.1,
                                                         max_shear=0.1,
                                                         min_scaling=(0.9, 0.9),
                                                         max_scaling=(1.1, 1.1),
                                                         flip_x_chance=0.5,
                                                         flip_y_chance=0.5,
                                                         )
        visual_effect_generator = random_visual_effect_generator(contrast_range=(0.9, 1.1),
                                                                 brightness_range=(-.1, .1),
                                                                 hue_range=(-0.05, 0.05),
                                                                 saturation_range=(0.95, 1.05)
                                                                 )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
        visual_effect_generator = None

    train_generator = TextGenerator(annotations, classes, transform_generator=transform_generator,
                                    visual_effect_generator=visual_effect_generator, **common_args)

    if val_annotations:
        validation_generator = TextGenerator(val_annotations, classes, shuffle_groups=False, **common_args)
    else:
        validation_generator = None

    return train_generator, validation_generator


def train(batch_size=32, backbone='resnet50', annotations=None, classes=None):
    backbone = models.backbone(backbone)
    train_generator, validation_generator = create_generators(backbone.preprocess_image, batch_size=batch_size,
                                                              annotations=annotations,
                                                              classes=classes)
    model, training_model, prediction_model = create_models(backbone_retinanet=backbone.retinanet,
                                                            num_classes=train_generator.num_classes(),
                                                            freeze_backbone=False,
                                                            lr=1e-5, )

    print(model.summary())

    callbacks = create_callbacks(batch_size, tensorboard_dir=None)

    return training_model.fit_generator(generator=train_generator,
                                        steps_per_epoch=100,
                                        epochs=50,
                                        verbose=1,
                                        callbacks=callbacks,
                                        use_multiprocessing=False,
                                        validation_data=validation_generator,
                                        )


if __name__ == '__main__':
    annotations = '/Users/liangtaoniu/workspace/MyDiary/src/tianchi/live/data/train/image_item_train.txt'
    classes = '/Users/liangtaoniu/workspace/MyDiary/src/tianchi/live/data/classes/coco.names'
    train(annotations=annotations, classes=classes, batch_size=4)
