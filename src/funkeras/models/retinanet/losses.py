from typing import Any, Callable

import tensorflow as tf
from tensorflow.keras import backend as K


def focal(alpha: float = 0.25, gamma: float = 2.0) -> Callable[[Any, Any], Any]:
    """构造计算 Focal Loss 的损失函数（参见 https://arxiv.org/abs/1708.02002）。

    参数:
        alpha: 正负样本的加权系数。
        gamma: 难易样本的加权指数。

    返回:
        接收 ``(y_true, y_pred)`` 并返回 focal loss 标量张量的可调用对象。
    """

    def _focal(y_true: Any, y_pred: Any) -> Any:
        """计算单个 batch 的 focal loss。

        参数:
            y_true: 形状 ``(B, N, num_classes + 1)`` 的目标张量，最后一维为
                anchor 状态（-1 忽略，0 背景，1 目标）。
            y_pred: 形状 ``(B, N, num_classes)`` 的网络预测张量。

        返回:
            对所有正样本 anchor 归一化后的 focal loss 标量。
        """
        labels = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(K.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = K.ones_like(labels) * alpha
        alpha_factor = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(K.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * K.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(K.equal(anchor_state, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

        return K.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma: float = 3.0) -> Callable[[Any, Any], Any]:
    """构造 Smooth L1 损失函数。

    参数:
        sigma: 控制损失从 L2 切换到 L1 的分界点。

    返回:
        接收 ``(y_true, y_pred)`` 并返回 smooth L1 loss 标量张量的可调用对象。
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true: Any, y_pred: Any) -> Any:
        """计算单个 batch 的 smooth L1 loss。

        参数:
            y_true: 形状 ``(B, N, 5)`` 的目标张量，最后一维为 anchor 状态
                （忽略/负样本/正样本）。
            y_pred: 形状 ``(B, N, 4)`` 的网络回归预测张量。

        返回:
            对所有正样本 anchor 归一化后的 smooth L1 loss 标量。
        """
        # separate target and state
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices = tf.where(K.equal(anchor_state, 1))
        regression = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = K.abs(regression_diff)
        regression_loss = tf.where(
            K.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * K.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = K.maximum(1, K.shape(indices)[0])
        normalizer = K.cast(normalizer, dtype=K.floatx())
        return K.sum(regression_loss) / normalizer

    return _smooth_l1
