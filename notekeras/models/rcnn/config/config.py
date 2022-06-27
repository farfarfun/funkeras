import math


class Config:

    def __init__(self):
        # Print the process or not
        self.verbose = True

        # Name of base network
        self.network = 'vgg'

        # Setting for data augmentation
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False

        # Anchor box scales
        # Note that if im_size is smaller, anchor_box_scales should be scaled
        # Original anchor_box_scales in the paper is [128, 256, 512]
        self.anchor_box_scales = [64, 128, 256]

        # Anchor box ratios
        self.anchor_box_ratios = [[1, 1], [1. / math.sqrt(2), 2. / math.sqrt(2)],
                                  [2. / math.sqrt(2), 1. / math.sqrt(2)]]

        # Size to resize the smallest side of the image
        # Original setting in paper is 600. Set to 300 in here to save training time
        self.im_size = 300

        # image channel-wise mean to subtract
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        # number of ROIs at once
        self.num_rois = 4

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 16

        self.balanced_classes = False

        # scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # TODO
        # should be replace with your own dataset's mapping
        # with bg as the number of classes
        # for example, if your dataset has three classes
        # your class_mapping should go like this {'class1': 0, 'class2': 1, 'class3': 2, 'bg':3}
        classes = open('/Users/liangtaoniu/workspace/MyDiary/src/tianchi/live/data/classes/coco.names', 'r').read()
        classes = classes.split('\n')
        class_map = {}
        for i, key in enumerate(classes):
            class_map[key] = i
        self.class_mapping = {'pikachu': 0, 'bg': 1}
        self.class_mapping = class_map

        self.model_path = None
        self.training_annotation = None
        self.cfg_save_path = None
        self.classes_count = None
