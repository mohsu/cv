# -*- coding:utf-8 _*-  
"""
@author: Maureen Hsu
@file: layers.py 
@time: 2020/06/16
"""

# python packages

# 3rd-party packages
from loguru import logger
from tensorflow.keras.layers import (
    Dense,
    Conv2D, SeparableConv2D, DepthwiseConv2D,
    Add, Concatenate, Reshape, multiply, Flatten,
    GlobalAveragePooling2D, Activation)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

# self-defined packages
from cv.module.utils import compose

kernel_reg = l2(5e-5)  # regularizer weight

"""
Elementary Layer Definition
"""


def Conv2DBasic(*args, **kwargs):
    default_kwargs = {'strides': (1, 1),
                      'kernel_regularizer': kernel_reg,
                      'padding': 'same',
                      'kernel_initializer': 'he_uniform'}
    default_kwargs.update(kwargs)
    return Conv2D(*args, **default_kwargs)


def SeparableConvBasic(*args, **kwargs):
    default_kwargs = {'strides': (1, 1),
                      'pointwise_regularizer': kernel_reg,
                      'padding': 'same',
                      'depthwise_initializer': 'he_uniform'}
    default_kwargs.update(kwargs)
    return SeparableConv2D(*args, **default_kwargs)


def DepthwiseConvBasic(*args, **kwargs):
    default_kwargs = {'strides': (1, 1),
                      'depthwise_regularizer': kernel_reg,
                      'padding': 'same',
                      'depthwise_initializer': 'he_uniform'}
    default_kwargs.update(kwargs)
    return DepthwiseConv2D(*args, **default_kwargs)


"""
Combined Layer Definition
"""


def ConvolutionBN(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        Conv2DBasic(*args, **kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def SeparableConvolutionBN(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        SeparableConvBasic(*args, **kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def DepthwiseConvolutionBNLayer(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DepthwiseConvBasic(*args, **kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


"""
SOTA module Definition
"""


def mobilenet_v2_reduce_body(input, out_channels, expansion_factor=2):
    in_channels = input.get_shape().as_list()[3]

    x = ConvolutionBN(in_channels * expansion_factor, kernel_size=(1, 1))(input)
    x = DepthwiseConvolutionBNLayer(kernel_size=(3, 3), strides=(2, 2))(x)
    x = Conv2DBasic(out_channels, kernel_size=(1, 1))(x)
    return x


def mobilenet_v2_body(input, expansion_factor=2):
    in_channels = input.get_shape().as_list()[3]

    x = ConvolutionBN(in_channels * expansion_factor, kernel_size=(1, 1))(input)
    x = DepthwiseConvolutionBNLayer(kernel_size=(3, 3))(x)
    x = Conv2DBasic(in_channels, kernel_size=(1, 1))(x)
    x = Add()([x, input])
    return x


def RecognizeLayer(x):
    x_number = Dense(128)(x)
    x_number = BatchNormalization()(x_number)
    x_number = LeakyReLU(alpha=0.1)(x_number)
    x_number = Dense(7)(x_number)
    return x_number


def ColorLayer(x):
    x = GlobalAveragePooling2D()(x)
    x_color = Dense(128)(x)
    x_color = BatchNormalization()(x_color)
    x_color = LeakyReLU(alpha=0.1)(x_color)
    x_color = Dense(2, activation='softmax', name="color")(x_color)
    return x_color


def AttentionRoiLayer(_input, x_color, ratio=16):
    conv_shape = _input.get_shape().as_list()
    h, w, filters = conv_shape[1:]

    x = ConvolutionBN(filters=filters // ratio, kernel_size=(1, 1))(_input)
    x = Flatten()(x)
    x = Concatenate(axis=-1)([x, x_color])
    x = Dense(h * w * 2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(h * w)(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Reshape((h, w, 1))(x)

    x = multiply([_input, x])
    return x


def AttentionLayer(_input, ratio=16):
    """
    SE Net: global pooling -> fc -> relu -> fc -> sigmoid ---|
                |                                            |
                --------------------------------------------->  scale
    :param input:
    :param ratio:
    :return:
    """
    conv_shape = _input.get_shape().as_list()
    filters = conv_shape[-1]

    x = GlobalAveragePooling2D()(_input)
    x = Reshape((1, 1, filters))(x)
    x = Dense(filters // ratio)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(filters)(x)
    x = Activation('sigmoid')(x)
    x = multiply([_input, x])
    return x
