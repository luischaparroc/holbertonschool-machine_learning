#!/usr/bin/env python3
"""Dropout create layer"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Dropout create layer"""
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(rate=keep_prob)
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=k_init,
        kernel_regularizer=dropout,
        activation=activation,
    )
    y = layer(prev)
    return y
