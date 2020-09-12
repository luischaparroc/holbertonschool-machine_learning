#!/usr/bin/env python3
"""LeNet5"""
import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of the LeNet-5 architecture using
    keras"""

    init = K.initializers.HeNormal()
    activation = K.activations.relu

    conv2d1 = K.layers.Conv2D(
        6,
        (5, 5),
        activation=activation,
        padding='same',
        kernel_initializer=init
    )(X)

    maxpool1 = K.layers.MaxPooling2D(
        (2, 2),
        (2, 2),
    )(conv2d1)

    con2d2 = K.layers.Conv2D(
        16,
        (5, 5),
        activation=activation,
        padding='valid',
        kernel_initializer=init
    )(maxpool1)

    maxpool2 = K.layers.MaxPooling2D(
        (2, 2),
        (2, 2),
    )(con2d2)

    flatten = K.layers.Flatten()(maxpool2)

    fullcc1 = K.layers.Dense(
        120,
        activation=activation,
        kernel_initializer=init
    )(flatten)

    fullcc2 = K.layers.Dense(
        84,
        activation=activation,
        kernel_initializer=init
    )(fullcc1)

    fullcc3 = K.layers.Dense(
        10,
        kernel_initializer=init
    )(fullcc2)

    output = K.activations.softmax(fullcc3)

    model = K.Model(inputs=X, outputs=output)

    return model
