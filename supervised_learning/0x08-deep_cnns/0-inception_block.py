#!/usr/bin/env python3
"""Inception block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)

    convF1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        activation='relu',
        padding='same',
        kernel_initializer=init
    )(A_prev)

    convF3R = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        activation='relu',
        padding='same',
        kernel_initializer=init
    )(A_prev)

    convF3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_initializer=init
    )(convF3R)

    convF5R = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        activation='relu',
        padding='same',
        kernel_initializer=init
    )(A_prev)

    convF5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        activation='relu',
        padding='same',
        kernel_initializer=init
    )(convF5R)

    maxpool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same',
    )(A_prev)

    convFPP = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(maxpool)

    output = K.layers.concatenate([convF1, convF3, convF5, convFPP])

    return output
