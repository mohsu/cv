# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: models.py 
@time: 2020/05/22
"""

# python packages
from absl import flags

# 3rd-party packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Lambda,
    MaxPool2D,
    Flatten
)

# self-defined packages
from cv.yolov3.core.layers import (
    DarknetConv,
    DarknetBlock,
    YoloConv,
    YoloConvTiny,
    YoloOutput,
    yolo_boxes,
    yolo_nms,
    stack_preds,
    multiclass_nms
)

flags.DEFINE_integer('yolo_max_boxes', 10, 'maximum number of boxes per image')
flags.DEFINE_float('yolo_iou_threshold', 0.45, 'nms iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.2, 'nms score threshold')
flags.DEFINE_float('yolo_soft_nms_sigma', 0.0, 'soft_nms_sigma')
flags.DEFINE_float('yolo_label_smoothing', 0.0, 'label smoothing alpha')
flags.DEFINE_float('yolo_scale_xy', 1.1, 'yolo_scale_xy', 1.0, None)

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169), (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def get_anchor_masks(anchors):
    if len(anchors) // 3 == 3:
        return yolo_anchor_masks
    else:
        return yolo_tiny_anchor_masks


def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = Lambda(lambda _input: _input / 255.0)(x)
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return Model(inputs, (x_36, x_61, x), name=name)


def YoloV3(size=None, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=[80], training=False):
    if isinstance(size, tuple):
        input_shape = size + (channels,)
    else:
        input_shape = (size, size, channels)

    x = inputs = Input(input_shape, name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name='yolo_nms')(
        (boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def DarknetTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = Lambda(lambda _input: _input / 255.0)(x)
    x = DarknetConv(x, 16, 3, mish_activation=True)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3, mish_activation=True)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3, mish_activation=True)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3, mish_activation=True)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3, mish_activation=True)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3, mish_activation=True)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3, mish_activation=True)
    return Model(inputs, (x_8, x), name=name)


def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=[80], training=False):
    if isinstance(size, tuple):
        input_shape = size + (channels,)
    else:
        input_shape = (size, size, channels)

    x = inputs = Input(input_shape, name='input')

    x_8, x = DarknetTiny(name='yolo_darknet')(x)

    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3_tiny')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')


def YoloCustomizeModel(size=None, channels=3,
                       anchors=yolo_anchors, masks=None,
                       classes=80, training=False,
                       model=None,
                       normalize_input=True):
    """
    :param size:
    :param channels:
    :param anchors:
    :param masks:
    :param classes:
    :param training:
    :param model:
    :param normalize_input:
    :return:
    """
    # num_layer=2 --> YoloV3Tiny, num_layer=3 ---> YoloV3
    num_layer = len(anchors) // 3
    tiny = True if (num_layer == 2) else False
    if masks is None:
        masks = get_anchor_masks(anchors)

    if isinstance(size, tuple):
        input_shape = size + (channels,)
    else:
        input_shape = (size, size, channels)

    x = inputs = Input(input_shape, name='input')
    if normalize_input:
        x = Lambda(lambda _input: _input / 255.0)(x)

    if tiny:
        if model is None:
            x_skip, x = DarknetTiny(name='yolo_darknet')(x)
            x = y1 = YoloConvTiny(256, name='yolo_conv_0')(x)
            x = y2 = YoloConvTiny(128, name='yolo_conv_1')((x, x_skip))
        else:
            y1, y2 = model(x)

        output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(y1)
        output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(y2)

        outputs = [output_0, output_1]
    else:
        x_36, x_61, x = Darknet(name='yolo_darknet')(x)

        x = YoloConv(512, name='yolo_conv_0')(x)
        output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

        x = YoloConv(256, name='yolo_conv_1')((x, x_61))
        output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

        x = YoloConv(128, name='yolo_conv_2')((x, x_36))
        output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

        outputs = [output_0, output_1, output_2]

    if training:
        return Model(inputs, outputs, name='yolov3')

    boxes = [Lambda(lambda x: yolo_boxes(x, anchors[masks[i]], classes), name=f'yolo_boxes_{i}')(outputs[i]) for i in
             range(num_layer)]
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name="yolo_nms")((box[:3] for box in boxes))

    return Model(inputs, outputs, name='yolov3_tiny')


def YoloCustomizeModelMultiCategory(size=None, channels=3,
                                    anchors=yolo_anchors, classes=[80], masks=None, training=False,
                                    model=None, normalize_input=True, min_box=None):
    """
    :param size:
    :param channels:
    :param anchors:
    :param masks:
    :param classes:
    :param training:
    :param model:
    :param normalize_input:
    :return:
    """
    # num_layer=2 --> YoloV3Tiny, num_layer=3 ---> YoloV3
    num_layer = len(anchors) // 3
    tiny = True if (num_layer == 2) else False
    if masks is None:
        masks = get_anchor_masks(anchors)

    if isinstance(size, tuple):
        input_shape = size + (channels,)
    else:
        input_shape = (size, size, channels)

    x = inputs = Input(input_shape, name='input')
    if normalize_input:
        x = Lambda(lambda _input: _input / 255.0)(x)

    if tiny:
        if model is None:
            x_skip, x = DarknetTiny(name='yolo_darknet')(x)
            x = y1 = YoloConvTiny(256, name='yolo_conv_0')(x)
            y2 = YoloConvTiny(128, name='yolo_conv_1')((x, x_skip))
        else:
            y1, y2 = model(x)

        output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(y1)
        output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(y2)
        outputs = [output_0, output_1]

    else:
        x_36, x_61, x = Darknet(name='yolo_darknet')(x)

        x = y1 = YoloConv(512, name='yolo_conv_0')(x)
        x = y2 = YoloConv(256, name='yolo_conv_1')((x, x_61))
        y3 = YoloConv(128, name='yolo_conv_2')((x, x_36))

        output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(y1)
        output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(y2)
        output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(y3)
        outputs = [output_0, output_1, output_2]

    if training:
        return Model(inputs, outputs, name='yolov3')

    if min_box is not None:
        min_box = min_box / size
    boxes = [Lambda(lambda x: yolo_boxes(x, anchors[masks[i]], classes),
                    name=f'yolo_boxes_{i}')(outputs[i]) for i in range(num_layer)]
    preds = Lambda(lambda x: stack_preds(x, classes), name="stack_pred")((box[:3] for box in boxes))
    preds = Lambda(lambda x: multiclass_nms(x, min_box), name="nms")(preds)

    return Model(inputs, preds, name='yolov3')
