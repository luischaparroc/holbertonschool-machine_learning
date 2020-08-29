#!/usr/bin/env python3
"""l2 reg create layer"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a layer that includes L2 regularization"""
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=k_init,
        kernel_regularizer=regularizer,
        activation=activation,
        name='Layer'
    )
    y = layer(prev)
    return y
