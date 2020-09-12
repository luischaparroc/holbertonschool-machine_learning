#!/usr/bin/env python3
"""LeNet5"""
import tensorflow as tf


def lenet5(x, y):
    """Builds a modified version of the LeNet-5 architecture using
    tensorflow"""

    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    conv2d1 = tf.layers.Conv2D(
        6,
        (5, 5),
        activation=activation,
        padding='same',
        kernel_initializer=init
    )(x)

    maxpool1 = tf.layers.MaxPooling2D(
        (2, 2),
        (2, 2),
    )(conv2d1)

    con2d2 = tf.layers.Conv2D(
        16,
        (5, 5),
        activation=activation,
        padding='valid',
        kernel_initializer=init
    )(maxpool1)

    maxpool2 = tf.layers.MaxPooling2D(
        (2, 2),
        (2, 2),
    )(con2d2)

    flatten = tf.layers.Flatten()(maxpool2)

    fullcc1 = tf.layers.Dense(
        120,
        activation=activation,
        kernel_initializer=init
    )(flatten)

    fullcc2 = tf.layers.Dense(
        84,
        activation=activation,
        kernel_initializer=init
    )(fullcc1)

    fullcc3 = tf.layers.Dense(
        10,
        kernel_initializer=init
    )(fullcc2)

    loss = tf.losses.softmax_cross_entropy(y, fullcc3)

    train = tf.train.AdamOptimizer().minimize(loss)

    prediction = tf.equal(
        tf.argmax(fullcc3, 1),
        tf.argmax(y, 1)
    )

    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    output = tf.nn.softmax(fullcc3)

    return output, train, loss, accuracy
