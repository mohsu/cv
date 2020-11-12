# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: utils.py 
@time: 2020/05/25
"""

# python packages
import tensorflow as tf

# 3rd-party packages
import numpy as np
from loguru import logger
from numba import jit

# self-defined packages
from utils.flag_utils import get_flag


@logger.catch(reraise=True)
def broadcast_iou(box_1, box_2, DIoU=False, CIoU=False):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h

    h1 = box_1[..., 3] - box_1[..., 1]
    w1 = box_1[..., 2] - box_1[..., 0]
    h2 = box_2[..., 3] - box_2[..., 1]
    w2 = box_2[..., 2] - box_2[..., 0]
    box_1_area = h1 * w1
    box_2_area = h2 * w2

    iou = tf.math.divide_no_nan(int_area, box_1_area + box_2_area - int_area)

    return iou


def _get_v(h2, w2, h1, w1):
    """
    https://github.com/tensorflow/addons/pull/914
    :param h2: true_h
    :param w2: true_w
    :param h1: pred_h
    :param w1: pred_w
    :return:
    """

    @tf.custom_gradient
    def _get_grad_v(height, width):
        arctan = tf.atan(tf.math.divide_no_nan(w2, h2)) - tf.atan(
            tf.math.divide_no_nan(width, height)
        )
        v = 4 * ((arctan / np.pi) ** 2)

        def _grad_v(dv):
            gdw = dv * 8 * arctan * height / (np.pi ** 2)
            gdh = -dv * 8 * arctan * width / (np.pi ** 2)
            return [gdh, gdw]

        return v, _grad_v

    return _get_grad_v(h1, w1)


def calculate_iou(box_1, box_2, DIoU=False, CIoU=False, broadcast=True):
    """
    https://arxiv.org/pdf/1911.08287.pdf
    :param box_1: predicted bbox
    :param box_2: true_bbox
    :param DIoU: bool option for returning DIoU
    :param CIoU: bool option for returning CIoU
    :param broadcast: bool option if broadcast to same shape required
    :return: iou or diou or ciou
    """
    if broadcast:
        # box_1: (..., (x1, y1, x2, y2))
        # box_2: (N, (x1, y1, x2, y2))
        box_1 = tf.expand_dims(box_1, -2)
        box_2 = tf.expand_dims(box_2, 0)
        # new_shape: (..., N, (x1, y1, x2, y2))
        new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
        box_1 = tf.broadcast_to(box_1, new_shape)
        box_2 = tf.broadcast_to(box_2, new_shape)

    # box_1(box_pred): (batch_size, grid, grid, anchor, (x1, y1, x2, y2))
    # box_2(box_true): (batch_size, grid, grid, anchor, (x1, y1, x2, y2))
    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h

    h1 = box_1[..., 3] - box_1[..., 1]
    w1 = box_1[..., 2] - box_1[..., 0]
    h2 = box_2[..., 3] - box_2[..., 1]
    w2 = box_2[..., 2] - box_2[..., 0]

    box_1_area = h1 * w1
    box_2_area = h2 * w2

    iou = tf.math.divide_no_nan(int_area, box_1_area + box_2_area - int_area)

    if DIoU or CIoU:
        # outer diagonal
        enclose_left_up = tf.minimum(box_1[..., :2], box_2[..., :2])
        enclose_right_down = tf.maximum(box_1[..., 2:], box_2[..., 2:])
        enclose_section = enclose_right_down - enclose_left_up

        outer_diag = tf.square(enclose_section[..., 0]) + tf.square(enclose_section[..., 1])

        # inner diagonal
        center_diagonal = (box_1[..., :2] + box_1[..., 2:]) / 2 - (box_2[..., :2] + box_2[..., 2:]) / 2

        inter_diag = tf.square(center_diagonal[..., 0]) + tf.square(center_diagonal[..., 1])
        if DIoU or CIoU:
            # diou
            iou = iou - tf.math.divide_no_nan(inter_diag, outer_diag)
            if CIoU:
                v = _get_v(h2, w2, h1, w1)
                alpha = tf.math.divide_no_nan(v, ((1 - iou) + v))
                iou = iou - alpha * v  # diou - alpha * v
            iou = tf.clip_by_value(iou, -1.0, 1.0)
    return iou


@logger.catch(reraise=True)
def freeze_all(model, frozen=True, until_layer=0):
    if isinstance(model, tf.keras.Model):
        if until_layer < 0:
            until_layer = len(model.layers) + until_layer
        for i, l in enumerate(model.layers):
            if i >= until_layer:
                break
            freeze_all(l, frozen)
    else:
        model.trainable = not frozen


@logger.catch(reraise=True)
def yolo_decode_predictions(model_output, output_images):
    boxes, scores, classes, nums = model_output

    decoded = []
    for i in range(len(output_images)):  # batch_size
        wh = np.flip(output_images[i].shape[0:2])
        objs = []
        for n in range(nums[i]):
            x1y1 = tuple((np.array(boxes[i][n][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][n][2:4]) * wh).astype(np.int32))
            score = float(scores[i][n])
            label = int(classes[i][n])
            objs.append([x1y1 + x2y2, score, label])
        decoded.append(sorted(objs, key=lambda x: (x[0], x[1], x[-1])))
    return decoded


def decode_predictions(model_output, output_images):
    # boxes, scores, classes = model_output
    decoded = []
    for i in range(len(output_images)):  # batch_size
        wh = np.flip(output_images[i].shape[0:2])
        objs = []
        for box, scores, klasses in zip(*model_output[i]):
            x1y1 = tuple((np.array(box[0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(box[2:4]) * wh).astype(np.int32))
            objs.append([x1y1 + x2y2, scores, klasses])
        decoded.append(objs)
    return decoded


def rescale_pred_wh(pred_boxes, output_image_shape):
    wh = np.flip(output_image_shape[0:2])
    rescaled = np.array(pred_boxes * np.repeat(wh, 2)).astype(np.int32)
    return rescaled


def convert2_class(scores, classes):
    if isinstance(scores, list) or scores.ndim == 3:
        return [convert2_class(score, classes) for score in scores]
    scores = tf.split(scores, classes, axis=-1)
    selected_scores, selected_classes = [], []
    for c in range(len(classes)):
        selected_class = tf.argmax(scores[c], axis=-1)  # at least one dim
        ind = tf.cast(tf.expand_dims(selected_class, 1), tf.int32)
        ran = tf.expand_dims(tf.range(tf.shape(selected_class)[0], dtype=tf.int32), 1)
        ind = tf.concat([ran, ind], axis=1)
        selected_score = tf.gather_nd(scores[c], ind)
        selected_classes.append(selected_class)
        selected_scores.append(selected_score)

    return np.array(tf.stack(selected_classes, axis=-1)), np.array(tf.stack(selected_scores, axis=-1))


def decode_tf_nms(preds, test_image_shape=None):
    boxes, scores, nums = preds
    boxes = tf.map_fn(lambda x: x[0][:x[1]], elems=(boxes, nums), dtype=tf.float32)
    scores = tf.map_fn(lambda x: x[0][:x[1]], elems=(scores, nums), dtype=tf.float32)
    if test_image_shape:
        boxes = rescale_pred_wh(boxes, test_image_shape)
    # boxes = tf.split(boxes, nums)
    # scores = tf.split(scores, nums)

    boxes = [box for box in boxes]
    scores = [score.numpy() for score in scores]

    return boxes, scores


def trim_zeros_2d(arr_2d, axis=1):
    mask = ~(arr_2d == 0).all(axis=axis)
    return arr_2d[mask]


def decode_labels(test_labels, output_images):
    decoded = []

    def multiply_scale(x, scale):
        return int(float(x) * scale)

    for i in range(len(output_images)):
        labels = []
        H, W = output_images[i].shape[:2]
        for obj in test_labels[i]:
            xmin, ymin, xmax, ymax = obj[:4]
            klass = np.argmax(obj[4:])
            if sum(obj[:4]) != 0:  # exclude zero padding
                labels.append([(multiply_scale(xmin, W), multiply_scale(ymin, H),
                                multiply_scale(xmax, W), multiply_scale(ymax, H)),
                               klass])
        decoded.append(sorted(labels, key=lambda x: (x[0], x[1], x[-1])))
    return decoded


def find_closet_box(pred, objs, thres=20):
    coords, score, klass = pred

    losses = []
    for i, obj in enumerate(objs):
        if klass != obj[-1]:
            continue
        loss = sum(np.abs(np.array(coords) - np.array(obj[0])))
        if loss < thres:
            losses.append([i, loss])
    if not losses:
        return None, None
    losses = sorted(losses, key=lambda x: x[1])
    return losses[0]


@jit(nopython=True)
def non_max_suppression(boxes, scores, threshold):
    """
    While there are any remaining boxes:
        - Pick the box with largest score, output that as a prediction
        - Discard any remaining box with IoU >= IoU_threshold with the box output in the previous step
    :param boxes: [[x1, y1, x2, y2], ...]
    :param scores: [score1, score2]
    :param threshold: float
    :return: array of picked index of boxes
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (y2 - y1) * (x2 - x1)
    idxs = scores.ravel().argsort()

    pick = []
    while len(idxs) > 0:
        # 選擇score最大bounding box加入到候選隊列
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[:last]]  # interaction / area

        mask = np.zeros(idxs.shape[0]) == 0
        index = np.where(overlap > threshold)[0]
        mask[index] = False
        mask[last] = False
        idxs = idxs[mask]

    return np.array(pick)


def nms(preds, score_threshold=get_flag("yolo_score_threshold", 0.2),
        iou_threshold=get_flag("yolo_iou_threshold", 0.45),
        max_output=get_flag("yolo_max_boxes", 10), test_image_shape=None):
    boxes, scores_combined, scores = tf.split(preds, (4, 1, -1), axis=-1)
    boxes = rescale_pred_wh(boxes, test_image_shape)

    @jit(nopython=True)
    def _nms(boxes, scores_combined, scores):
        n_sample = boxes.shape[0]
        selected_boxes = []
        selected_scores = []
        # selected_boxes = np.zeros((max_output * n_sample, 4), dtype=np.float32)
        # selected_scores = np.zeros((max_output * n_sample, scores.shape[-1]), dtype=np.float32)
        # nums = np.zeros(n_sample, dtype=np.float32)

        idx_start = 0
        for n in range(n_sample):
            box, score_combined, score = boxes[n], scores_combined[n], scores[n]
            mask = (score_combined >= score_threshold).ravel()

            box = box[mask]
            score_combined = score_combined[mask]
            score = score[mask]
            nms_indexes = non_max_suppression(box, score_combined, iou_threshold)[:max_output]
            if len(nms_indexes) != 0:
                selected_boxes.append(box[nms_indexes])
                selected_scores.append(score[nms_indexes])
            else:
                selected_boxes.append(np.empty((0, box.shape[-1]), np.int32))
                selected_scores.append(np.empty((0, score.shape[-1]), np.float32))
            # num = len(nms_indexes)

            # if num != 0:
            #
            #     selected_boxes[idx_start: idx_start + num] = box[nms_indexes]
            #     selected_scores[idx_start: idx_start + num] = score[nms_indexes]
            # nums[n] = num

            # idx_start += num
        return selected_boxes, selected_scores

    return _nms(np.array(boxes), np.array(scores_combined), np.array(scores))
    boxes, scores, nums = _nms(np.array(boxes), np.array(scores_combined), np.array(scores))
    return tf.convert_to_tensor(boxes, tf.float32), tf.convert_to_tensor(scores, tf.float32), tf.convert_to_tensor(nums,
                                                                                                                   tf.int32)
