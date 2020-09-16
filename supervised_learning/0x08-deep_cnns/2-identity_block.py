#!/usr/bin/env python3
"""Identity block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Builds an identity block"""

    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=None)

    convF11 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        activation='relu',
        padding='same',
        kernel_initializer=init
    )(A_prev)

    bn1 = K.layers.BatchNormalization(
        axis=3
    )(convF11)

    activation1 = K.layers.Activation('relu')(bn1)

    convF3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        activation='relu',
        padding='same',
        kernel_initializer=init
    )(activation1)

    bn2 = K.layers.BatchNormalization(
        axis=3
    )(convF3)

    activation2 = K.layers.Activation('relu')(bn2)

    convF12 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        activation='relu',
        padding='same',
        kernel_initializer=init
    )(activation2)

    bn3 = K.layers.BatchNormalization(
        axis=3
    )(convF12)

    summation = K.layers.Add()([A_prev, bn3])

    activation3 = K.layers.Activation('relu')(summation)

    return activation3
