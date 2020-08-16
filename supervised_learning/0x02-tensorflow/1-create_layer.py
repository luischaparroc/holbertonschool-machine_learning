#!/usr/bin/env python3
"""Create layer module"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Function that creates a layer
    for a neural network
    """
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=k_init,
        activation=activation,
        name='Layer'
    )
    y = layer(prev)
    return y
