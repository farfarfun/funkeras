import pickle

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras import Model
from keras.layers import Input
from utils.image_processing import format_img, get_real_coordinates
from utils.nms import apply_regr, non_max_suppression_fast, rpn_to_roi


def init_config(net, weights):
    cfg_path = 'config/res_config.pickle' if net == 'res' else 'config/vgg_config.pickle'
    with open(cfg_path, 'rb') as f_in:
        cfg = pickle.load(f_in)
    cfg.use_horizontal_flips = False
    cfg.use_vertical_flips = False
    cfg.rot_90 = False
    cfg.model_path = weights
    return cfg


def test(net, weights, image_path, thresh=0.7):
    cfg = init_config(net, weights)
    if net == 'res':
        num_features = 1024
        from layers.resnet50 import nn_base, rpn_layer, classifier_layer
    else:
        num_features = 512
        from layers.vgg16 import nn_base, rpn_layer, classifier_layer

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(cfg.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn_layers = rpn_layer(shared_layers, num_anchors)

    classifier = classifier_layer(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(cfg.class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(cfg.model_path))
    model_rpn.load_weights(cfg.model_path, by_name=True)
    model_classifier.load_weights(cfg.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    # Switch key value for class mapping
    class_mapping = cfg.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}

    img = cv2.imread(image_path)

    X, ratio = format_img(img, cfg)
    X = np.transpose(X, (0, 2, 3, 1))
    [Y1, Y2, F] = model_rpn.predict(X)
    R = rpn_to_roi(Y1, Y2, cfg, 'tf', overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // cfg.num_rois + 1):
        ROIs = np.expand_dims(R[cfg.num_rois * jk:cfg.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // cfg.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        # Calculate bboxes coordinates on resized image
        for ii in range(P_cls.shape[1]):
            # Ignore 'bg' class
            if np.max(P_cls[0, ii, :]) < thresh or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append(
                [cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    show_result(image_path, bboxes, probs, ratio)


def show_result(image_path, bboxes, probs, ratio):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font='FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]
            (left, top, right, bottom) = get_real_coordinates(ratio, x1, y1, x2, y2)
            label = '{} {:.2f}'.format(key, new_probs[jk])

            # draw frame
            draw.rectangle(
                [left, top, right, bottom],
                outline=(255, 255, 255), width=thickness)

            # draw text
            label_size = draw.textsize(label, font)
            text_origin = np.array([left, top - label_size[1]])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=(255, 255, 255))
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw
    image.show()
    # image.save('results/pokemon_result_02.png', 'PNG')


if __name__ == '__main__':
    # test(net='vgg', weights='weights/tobacco_vgg.hdf5', image_path='testing_images/DJI_0082/DJI_0082-0-512.png')
    test(net='vgg', weights='weights/pokemon_vgg.hdf5', image_path='pokemon_test/8.jpg')
