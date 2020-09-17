#!/usr/bin/env python3
"""Transition Layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a Transition Layer"""

    init = K.initializers.he_normal(seed=None)

    bn1 = K.layers.BatchNormalization()(X)
    activation1 = K.layers.Activation('relu')(bn1)
    filters = int(nb_filters * compression)
    conv1 = K.layers.Conv2D(
        filters=filters,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(activation1)

    avgpool = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='same',
    )(conv1)

    return avgpool, filters
