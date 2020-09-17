#!/usr/bin/env python3
"""Dense block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a Dense block"""

    init = K.initializers.he_normal(seed=None)

    for _ in range(layers):
        bn1 = K.layers.BatchNormalization()(X)
        activation1 = K.layers.Activation('relu')(bn1)
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            kernel_initializer=init
        )(activation1)
        bn2 = K.layers.BatchNormalization()(conv1)
        activation2 = K.layers.Activation('relu')(bn2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=init
        )(activation2)
        X = K.layers.concatenate([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
