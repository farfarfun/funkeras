import pickle
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.utils import generic_utils

import notekeras.models.resnet as res
from notekeras.models.rcnn.config.config import Config
from notekeras.models.rcnn.layers import resnet50
from notekeras.models.rcnn.layers.loss import class_loss_cls, class_loss_regr, rpn_loss_regr, rpn_loss_cls
from notekeras.models.rcnn.utils import anchor
from notekeras.models.rcnn.utils import image_processing
from notekeras.models.rcnn.utils.iou import calc_iou
from notekeras.models.rcnn.utils.nms import rpn_to_roi

cfg = Config()
net = resnet50

cfg.network = resnet50
cfg.use_horizontal_flips = True
cfg.use_vertical_flips = True
cfg.rot_90 = True
cfg.base_net_weights = '/Users/liangtaoniu/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
cfg.cfg_save_path = '/Users/liangtaoniu/workspace/MyDiary/tmp/models/rcnn/config/res_config.pickle'
cfg.model_path = '/Users/liangtaoniu/workspace/MyDiary/tmp/models/rcnn/weights/res_frcnn.hdf5'

cfg.training_annotation = '/Users/liangtaoniu/workspace/MyDiary/src/tianchi/live/data/train/image_item_train.txt'


def get_data():
    training_images, classes_count, class_mapping = image_processing.get_data(cfg.training_annotation,
                                                                              cfg.class_mapping)
    cfg.class_mapping = class_mapping
    cfg.classes_count = classes_count

    with open(cfg.cfg_save_path, 'wb') as config_f:
        try:
            pickle.dump(cfg, config_f)
        except Exception as e:
            print(e)

    # Shuffle the images with seed
    random.seed(1)
    random.shuffle(training_images)

    # Get train data generator which generate X, Y, image_data
    data_gen_train = anchor.get_anchor_gt(training_images, cfg, net.get_img_output_length, mode='train')
    # X, Y, image_data, debug_img, debug_num_pos = next(data_gen_train)
    show_anchor(data_gen_train)
    return data_gen_train


def show_anchor(data_gen_train):
    X, Y, image_data, debug_img, debug_num_pos = next(data_gen_train)
    print('Original image: height=%d width=%d' % (image_data['height'], image_data['width']))
    print('Resized image:  height=%d width=%d C.im_size=%d' % (X.shape[1], X.shape[2], cfg.im_size))
    print('Feature map size: height=%d width=%d C.rpn_stride=%d' % (Y[0].shape[1], Y[0].shape[2], cfg.rpn_stride))
    print(X.shape)
    print(str(len(Y)) + " includes 'y_rpn_cls' and 'y_rpn_regr'")
    print('Shape of y_rpn_cls {}'.format(Y[0].shape))
    print('Shape of y_rpn_regr {}'.format(Y[1].shape))
    print(image_data)

    print('Number of positive anchors for this image: %d' % (debug_num_pos))
    if debug_num_pos == 0:
        gt_x1, gt_x2 = image_data['bboxes'][0]['x1'] * (X.shape[2] / image_data['height']), image_data['bboxes'][0][
            'x2'] * (X.shape[2] / image_data['height'])
        gt_y1, gt_y2 = image_data['bboxes'][0]['y1'] * (X.shape[1] / image_data['width']), image_data['bboxes'][0][
            'y2'] * (X.shape[1] / image_data['width'])
        gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

        img = debug_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = (0, 255, 0)
        cv2.putText(img, 'gt bbox', (gt_x1, gt_y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
        cv2.circle(img, (int((gt_x1 + gt_x2) / 2), int((gt_y1 + gt_y2) / 2)), 3, color, -1)

        plt.grid()
        plt.imshow(img)
        plt.show()
    else:
        cls = Y[0][0]
        pos_cls = np.where(cls == 1)
        print(pos_cls)
        regr = Y[1][0]
        pos_regr = np.where(regr == 1)
        print(pos_regr)
        print('y_rpn_cls for possible pos anchor: {}'.format(cls[pos_cls[0][0], pos_cls[1][0], :]))
        print('y_rpn_regr for positive anchor: {}'.format(regr[pos_regr[0][0], pos_regr[1][0], :]))

        gt_x1, gt_x2 = image_data['bboxes'][0]['x1'] * (X.shape[2] / image_data['width']), image_data['bboxes'][0][
            'x2'] * (X.shape[2] / image_data['width'])
        gt_y1, gt_y2 = image_data['bboxes'][0]['y1'] * (X.shape[1] / image_data['height']), image_data['bboxes'][0][
            'y2'] * (X.shape[1] / image_data['height'])
        gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

        img = debug_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = (0, 255, 0)
        #   cv2.putText(img, 'gt bbox', (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
        cv2.circle(img, (int((gt_x1 + gt_x2) / 2), int((gt_y1 + gt_y2) / 2)), 3, color, -1)

        # Add text
        textLabel = 'gt bbox'
        (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        textOrg = (gt_x1, gt_y1 + 5)
        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                      (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
        cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        # Draw positive anchors according to the y_rpn_regr
        for i in range(debug_num_pos):
            color = (100 + i * (155 / 4), 0, 100 + i * (155 / 4))

            idx = pos_regr[2][i * 4] / 4
            anchor_size = cfg.anchor_box_scales[int(idx / 3)]
            anchor_ratio = cfg.anchor_box_ratios[2 - int((idx + 1) % 3)]

            center = (pos_regr[1][i * 4] * cfg.rpn_stride, pos_regr[0][i * 4] * cfg.rpn_stride)
            print('Center position of positive anchor: ', center)
            cv2.circle(img, center, 3, color, -1)
            anc_w, anc_h = anchor_size * anchor_ratio[0], anchor_size * anchor_ratio[1]
            cv2.rectangle(img, (center[0] - int(anc_w / 2), center[1] - int(anc_h / 2)),
                          (center[0] + int(anc_w / 2), center[1] + int(anc_h / 2)), color, 2)
    #         cv2.putText(img, 'pos anchor bbox '+str(i+1), (center[0]-int(anc_w/2), center[1]-int(anc_h/2)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

    print('Green bboxes is ground-truth bbox. Others are positive anchors')
    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.imshow(img)
    plt.show()


def train(network):
    # init_cfg(network)
    data_gen_train = get_data()

    input_shape_img = (None, None, 3)
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    # shared_layers = resnet50.nn_base(img_input, trainable=True)

    shared_layers = res.ResNet50Component()(img_input)

    # define the RPN, built on the base layers
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)  # 9
    rpn = resnet50.rpn_layer(shared_layers, num_anchors)

    classifier = resnet50.classifier_layer(shared_layers, roi_input, cfg.num_rois, nb_classes=len(cfg.classes_count))

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    model_rpn.load_weights(cfg.base_net_weights, by_name=True)
    model_classifier.load_weights(cfg.base_net_weights, by_name=True)

    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[class_loss_cls, class_loss_regr(len(cfg.classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(cfg.classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    epoch_length = 1000
    num_epochs = 40
    iter_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    best_loss = np.Inf
    r_epochs = 0
    total_epochs = 40

    start_time = time.time()
    for epoch_num in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(r_epochs + 1, total_epochs))

        r_epochs += 1

        while True:
            try:
                if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    if mean_overlapping_bboxes == 0:
                        print(
                            'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
                X, Y, img_data, debug_img, debug_num_pos = next(data_gen_train)

                # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
                loss_rpn = model_rpn.train_on_batch(X, Y)

                # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
                P_rpn = model_rpn.predict_on_batch(X)

                # R: bboxes (shape=(300,4))
                # Convert rpn layer to roi bboxes
                R = rpn_to_roi(P_rpn[0], P_rpn[1], cfg, 'tf', use_regr=True, overlap_thresh=0.7, max_boxes=300)

                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
                # Y1: one hot code for bboxes from above => x_roi (X)
                # Y2: corresponding labels and corresponding gt bboxes
                X2, Y1, Y2, IouS = calc_iou(R, img_data, cfg, cfg.class_mapping)

                # If X2 is None means there are no matching bboxes
                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                # Find out the positive anchors and negative anchors
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
                if len(pos_samples) < cfg.num_rois // 2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois // 2, replace=False).tolist()

                # Randomly choose (num_rois - num_pos) neg samples
                try:
                    selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                            replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                            replace=True).tolist()

                # Save all the pos and neg samples in sel_samples
                sel_samples = selected_pos_samples + selected_neg_samples

                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num,
                               [('rpn_cls', np.mean(losses[:iter_num, 0])),
                                ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                ('final_cls', np.mean(losses[:iter_num, 2])),
                                ('final_regr', np.mean(losses[:iter_num, 3]))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if cfg.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if cfg.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(cfg.model_path)

                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                continue

    print('Training complete, exiting.')


if __name__ == '__main__':
    train(network='vgg')
