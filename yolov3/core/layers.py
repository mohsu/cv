# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: layers.py 
@time: 2020/05/22
"""

# python packages
from absl.flags import FLAGS

# 3rd-party packages
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    ZeroPadding2D,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Add,
    Input,
    UpSampling2D,
    Concatenate,
    Lambda,
    Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
)
from tensorflow.keras.models import Model

# self-defined packages
from cv.yolov3.core.utils import (
    broadcast_iou,
    calculate_iou
)
from cv.module.Activation import Mish
from utils.flag_utils import get_flag


def DarknetConv(x, filters, size, strides=1, batch_norm=True, mish_activation=False):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        if mish_activation:
            x = Activation('mish')(x)
        else:
            x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters=filters, size=3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, size=1)
        x = DarknetConv(x, filters * 2, size=3)
        x = DarknetConv(x, filters, size=1)
        x = DarknetConv(x, filters * 2, size=3)
        x = DarknetConv(x, filters, size=1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    total_class_length = sum(classes) + 5

    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * total_class_length, 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, total_class_length)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


# @tf.function
def calculate_category_loss(class_probs, classes):
    def _calculate_loss(i):
        num_category_class = tf.gather(classes, i)
        start_index = tf.cast(tf.reduce_sum(tf.slice(classes, [0], [i])), tf.int32)
        end_index = tf.cast(start_index + num_category_class, tf.int32)
        loss = tf.sigmoid(class_probs[..., start_index:end_index])
        return loss

    return _calculate_loss


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, sum(classes)), axis=-1)

    box_xy = tf.sigmoid(box_xy) * get_flag("yolo_scale_xy", 1.1)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    # shape = (batch_size, )
    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections


@tf.function
def multiclass_nms(pred, min_box=None):
    batch_boxes, batch_scores_combined, batch_scores = tf.split(pred, (4, 1, -1), axis=-1)

    # filter small boxes
    if min_box is not None:
        small_boxes = tf.logical_and((batch_boxes[..., 3] - batch_boxes[..., 1]) < min_box * 2,
                                    (batch_boxes[..., 2] - batch_boxes[..., 0]) < min_box * 2)
        batch_scores_combined = tf.where(tf.expand_dims(small_boxes, -1), tf.zeros_like(batch_scores_combined), batch_scores_combined)

    batch_size = tf.shape(batch_boxes)[0]
    boxes = tf.zeros([0, 4])
    scores = tf.zeros([0, tf.shape(batch_scores)[-1]])
    nums = tf.zeros([0], dtype=tf.int32)

    def cond(i, *args):
        return tf.less(i, batch_size)

    @tf.function
    def body(i, boxes, scores, nums):
        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
            tf.reshape(batch_boxes[i], (-1, 4)),
            tf.reshape(batch_scores_combined[i], (-1,)),
            max_output_size=get_flag("yolo_max_boxes", 10),
            iou_threshold=get_flag("yolo_iou_threshold", 0.45),
            score_threshold=get_flag("yolo_score_threshold", 0.2),
            soft_nms_sigma=get_flag("yolo_soft_nms_sigma", 0.0))
        selected_boxes = tf.gather(batch_boxes[i], selected_indices)
        selected_scores = tf.gather(batch_scores[i], selected_indices)

        boxes = tf.concat([boxes, selected_boxes], axis=0)
        scores = tf.concat([scores, selected_scores], axis=0)
        nums = tf.concat([nums, [tf.shape(selected_indices)[0]]], axis=0)

        return tf.add(i, 1), boxes, scores, nums

    i = tf.constant(0)
    i, boxes, scores, nums = tf.while_loop(cond, body, loop_vars=[i, boxes, scores, nums],
                                           shape_invariants=[i.get_shape(),
                                                             tf.TensorShape([None, 4]),
                                                             tf.TensorShape([None, None]),
                                                             tf.TensorShape([None])],
                                           back_prop=False, parallel_iterations=8)

    return boxes, scores, nums


def multiclass_nms_tensorarray(pred):
    batch_boxes, batch_scores_combined, batch_scores = tf.split(pred, (4, 1, -1), axis=-1)
    boxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False, clear_after_read=True)
    scores = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False, clear_after_read=True)
    nums = tf.TensorArray(tf.int32, size=0, dynamic_size=True, infer_shape=False, clear_after_read=True)

    # selected = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False, clear_after_read=True)

    def cond(i, boxes, scores, nums):
        return tf.less(i, tf.shape(batch_boxes)[0])

    def body(i, boxes, scores, nums):
        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
            tf.reshape(batch_boxes[i], (-1, 4)),
            tf.reshape(batch_scores_combined[i], (-1,)),
            max_output_size=get_flag("yolo_max_boxes", 10),
            iou_threshold=get_flag("yolo_iou_threshold", 0.45),
            score_threshold=get_flag("yolo_score_threshold", 0.2))
        selected_boxes = tf.gather(batch_boxes[i], selected_indices)
        selected_scores = tf.gather(batch_scores[i], selected_indices)

        boxes.write(i, selected_boxes)
        scores.write(i, selected_scores)
        nums.write(i, tf.shape(selected_indices)[0])

        # selected.write(i, tf.concat([boxes, scores], axis=-1))
        # selected.append([boxes, scores])
        return tf.add(i, 1), boxes, scores, nums

    i = tf.constant(0)
    i, boxes, scores, nums = tf.while_loop(cond, body, loop_vars=[i, boxes, scores, nums], back_prop=False)

    boxes = boxes.concat()
    scores = scores.concat()
    nums = nums.stack()

    return boxes, scores, nums


def stack_preds(outputs, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)
    scores = confidence * class_probs
    batch_size = tf.size(bbox[:, 0, 0])
    boxes = tf.reshape(bbox, (batch_size, -1, 4))
    scores = tf.reshape(scores, (batch_size, -1, tf.shape(scores)[-1]))
    scores_combined = tf.ones(tf.shape(scores)[:-1])
    start_idx = 0
    for num_category_class in classes:
        end_idx = start_idx + num_category_class
        scores_combined *= tf.reduce_max(scores[..., start_idx:end_idx], axis=2)
        start_idx = end_idx
    scores_combined = tf.expand_dims(scores_combined, axis=-1)
    preds = tf.concat((boxes, scores_combined, scores), axis=-1)

    return preds

    # for n in tf.range(batch_size):
    #     selected_indices, selected_scores = tf.image.non_max_suppression_padded(
    #         boxes[n], scores_combined[n],
    #         max_output_size=get_flag("yolo_max_boxes", 10),
    #         iou_threshold=get_flag("yolo_iou_threshold", 0.45),
    #         score_threshold=get_flag("yolo_score_threshold", 0.2))
    #     selected_boxes = tf.gather(boxes[n], selected_indices)
    #     selected_scores = tf.gather(scores[n], selected_indices)


def weighted_binary_crossentropy(weights=1):
    def _calculate_weighted_binary_crossentropy(labels, output, from_logits):
        """Calculate weighted binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=tf.float32)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.math.log(output / (1 - output))

        return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output)

    def _weighted_binary_crossentropy(y_true, y_pred, from_logits=False):
        bce = _calculate_weighted_binary_crossentropy(y_true, y_pred, from_logits)
        weighted_bac = bce * tf.convert_to_tensor(weights, tf.float32)
        return K.mean(weighted_bac, axis=-1)

    return _weighted_binary_crossentropy


def calculate_loss(obj_mask, true_class, pred_class):
    def _calculate_loss(i, j):
        return obj_mask * binary_crossentropy(true_class[..., i:j],
                                              pred_class[..., i:j],
                                              label_smoothing=get_flag("yolo_label_smoothing", 0.0))

    return _calculate_loss


def YoloLoss(anchors, classes=[80], ignore_thresh=0.5, weights=[1, 1, 1, 1]):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, classes)
        # pred_box: (batch_size, grid, grid, anchors, (x1, y1, x2, y2))
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class = tf.split(y_true, (4, 1, sum(classes)), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # if predict small, penalize a lot
        # pred_wh = tf.where(tf.reduce_any(pred_wh < 23, axis=4, keepdims=True), tf.ones_like(pred_wh) * 10000, pred_wh)

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
                  tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(
                calculate_iou(x[0], tf.boolean_mask(x[1], tf.cast(x[2], tf.bool)), broadcast=True), axis=-1),
            (pred_box, true_box, obj_mask), tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        ciou = calculate_iou(pred_box, true_box, DIoU=True, broadcast=False)

        # 5. calculate all losses
        ciou_loss = obj_mask * (1 - ciou)
        xy_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss

        class_loss = tf.zeros(tf.shape(obj_loss))
        start_idx = 0
        for num_category_class in classes:
            end_idx = start_idx + num_category_class
            class_loss += calculate_loss(obj_mask, true_class, pred_class)(start_idx, end_idx)
            start_idx = end_idx

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=(1, 2, 3)))
        # xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        # wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_mean(tf.reduce_sum(obj_loss, axis=(1, 2, 3)))
        class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=(1, 2, 3)))

        # selected but too small
        # small_loss = tf.where(tf.logical_and(best_iou >= ignore_thresh, tf.reduce_any(pred_wh < 0.036, axis=4)),
        #                       tf.ones_like(best_iou) * 0.01,
        #                       tf.zeros_like(best_iou))
        # small_loss = tf.reduce_mean(tf.reduce_sum(small_loss, axis=(1, 2, 3)))

        all_loss = ciou_loss + obj_loss + class_loss
        return all_loss

    return yolo_loss
