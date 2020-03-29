import time

import cv2
import numpy as np
from PIL import Image

from notekeras.model.yolo import YoloBody
from notekeras.utils import read_lines, draw_bbox

root = '/Users/liangtaoniu/workspace/MyDiary/notechats/notekeras/example/yolo'
classes = read_lines(root + "/data/classes/coco.names")


def get_anchors():
    anchors = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
    anchors = '1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875'
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


anchors = get_anchors()

yolo_body = YoloBody(anchors=anchors, num_classes=len(classes))
yolo_body.load_weights("/Users/liangtaoniu/workspace/MyDiary/tmp/models/yolo/configs/yolov3.h5", freeze_body=3)


def image_demo(image_path):
    original_image, boxes = yolo_body.predict_box(image_path)
    image = draw_bbox(original_image, boxes, classes=classes)
    image = Image.fromarray(image)
    image.show()


def video_demo(video_path):
    vid = cv2.VideoCapture(video_path)
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("No image!")
        prev_time = time.time()
        frame, boxes = yolo_body.predict_box(original_image=frame)
        curr_time = time.time()

        image = draw_bbox(frame, boxes, classes=classes)
        result = np.asarray(image)

        exec_time = curr_time - prev_time

        info = "time: %.2f ms" % (1000 * exec_time)
        cv2.putText(result, text=info, org=(50, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


image_demo(root + "/docs/kite.jpg")
# image_demo("'/Users/liangtaoniu/workspace/MyDiary/tmp/models/yolo/test/bb.jpg'")

video_demo(video_path=root + "/docs/road.mp4")
