#!/usr/bin/env python3
"""Create batch norm layer"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network
    in tensorflow
    """
    k_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    variance_epsilon = 1e-8
    layer_tanh = tf.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=k_init,
    )
    y = layer_tanh(prev)

    avg, var = tf.nn.moments(y, [0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    batch_normalization = tf.nn.batch_normalization(
        y,
        avg,
        var,
        beta,
        gamma,
        variance_epsilon,
    )

    layer_batch_normalization = activation(batch_normalization)

    return layer_batch_normalization
