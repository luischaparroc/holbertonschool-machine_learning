#!/usr/bin/env python3
"""LeNet5"""
import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of the LeNet-5 architecture using
    keras"""

    init = K.initializers.he_normal()

    conv2d1 = K.layers.Conv2D(
        6,
        (5, 5),
        activation='relu',
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
        activation='relu',
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
        activation='relu',
        kernel_initializer=init
    )(flatten)

    fullcc2 = K.layers.Dense(
        84,
        activation='relu',
        kernel_initializer=init
    )(fullcc1)

    fullcc3 = K.layers.Dense(
        10,
        activation='softmax',
        kernel_initializer=init
    )(fullcc2)

    model = K.Model(inputs=X, outputs=fullcc3)

    opt = K.optimizers.Adam()

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    return model
