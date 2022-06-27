import time

import cv2
import numpy as np
from PIL import Image
from notedrive.lanzou import download
from notekeras.model.yolo3 import YoloBody
from notekeras.utils import read_lines, draw_bbox

root = '/root/workspace/notechats/notekeras/example/yolo'
classes = read_lines(root + "/data/classes/coco.names")


download('https://wws.lanzous.com/b01hjn3aj', dir_pwd=root + '/models/')


def get_anchors():
    anchors = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
    # anchors = '1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875'
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


anchors = get_anchors()

yolo_body = YoloBody(anchors=anchors, num_classes=len(classes))
yolo_body.load_weights(root+"/models/yolov3.h5", freeze_body=3)


# yolo_body.load_layer_weights()


def image_demo(image_path, output):
    original_image, boxes = yolo_body.predict_box(image_path)
    image = draw_bbox(original_image, boxes, classes=classes)
    image = Image.fromarray(image)
    print(boxes)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(output, image)


def image_demos(image_path):
    image_path = [image_path, image_path, image_path,
                  image_path, image_path, image_path]
    boxes = yolo_body.predict_result_batch(image_path)
    print(boxes[0])


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


#image_demos(root + "/yolov3/docs/kite.jpg")
image_demo(root + "/data/images/kite.jpg",
           root + "/yolov3/results/kite-res.jpg")
