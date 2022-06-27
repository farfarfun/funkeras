import glob

import numpy as np

from notekeras.temp.RPN.utils import load_gt_boxes


def iou(box, clusters):
    """
    :param box:      np.array of shape (2,) containing w and h
    :param clusters: np.array of shape (N cluster, 2)
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def kmeans(boxes, k, dist=np.median, seed=1):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))  ## N row x N cluster
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)

    # initialize the cluster centers to be k items
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for icluster in range(k):  # I made change to lars76's code here to make the code faster
            distances[:, icluster] = 1 - iou(clusters[icluster], boxes)

        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def get_wh_from_boxes(boxes):
    """
    box shape: [-1, 4], return the width and height of boxes
    """
    return boxes[..., 2:4] - boxes[..., 0:2]


text_paths = glob.glob("./synthetic_dataset/imageAno/*.txt")
all_boxes = [load_gt_boxes(path) for path in text_paths]
all_boxes = np.vstack(all_boxes)
all_boxes_wh = get_wh_from_boxes(all_boxes)
anchors = kmeans(all_boxes_wh, k=9)
print(anchors)
