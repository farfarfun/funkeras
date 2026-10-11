"""使用 COCO 预训练 RetinaNet 检测单张图片。"""

import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from funkeras.models.retinanet import models
from funkeras.models.retinanet.models.retinanet import RetinaNetBox
from funkeras.models.retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from funkeras.models.retinanet.utils.visualization import draw_box, draw_caption, label_color


LABELS_TO_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
    7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
    13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
    28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
    53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush',
}


def parse_args() -> argparse.Namespace:
    """读取模型权重和待检测图片路径。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default=os.environ.get("FUNKERAS_RETINANET_WEIGHTS"),
        help="COCO/训练所得的 .h5 权重路径；也可设置 FUNKERAS_RETINANET_WEIGHTS。",
    )
    parser.add_argument("--image", default="kite.jpg", help="待检测图片路径（默认：kite.jpg）。")
    args = parser.parse_args()
    if not args.weights:
        parser.error("请通过 --weights 或 FUNKERAS_RETINANET_WEIGHTS 提供 RetinaNet 权重路径")
    if not os.path.isfile(args.weights):
        parser.error("找不到 RetinaNet 权重文件：{}".format(args.weights))
    if not os.path.isfile(args.image):
        parser.error("找不到待检测图片：{}".format(args.image))
    return args


def main() -> None:
    """加载权重，对图片执行推理并展示检测框。"""
    args = parse_args()
    model = models.backbone("resnet50").retinanet(80, num_anchors=None, modifier=None)
    model.load_weights(args.weights, by_name=True, skip_mismatch=True)
    model = RetinaNetBox(model=model, anchor_params=None)

    image = read_image_bgr(args.image)
    draw = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    image, scale = resize_image(preprocess_image(image))

    start = time.time()
    boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    boxes /= scale

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score < 0.5:
            break
        bounding_box = box.astype(int)
        draw_box(draw, bounding_box, color=label_color(label))
        draw_caption(draw, bounding_box, "{} {:.3f}".format(LABELS_TO_NAMES[label], score))

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.imshow(draw)
    plt.show()


if __name__ == "__main__":
    main()
