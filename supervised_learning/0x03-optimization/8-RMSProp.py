#!/usr/bin/env python3
"""Create RMSProp op"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network,, using
    the RMSProp optimization algorithm
    """
    train = tf.train.RMSPropOptimizer(
        alpha,
        decay=beta2,
        epsilon=epsilon
    )
    grads_and_vars = train.compute_gradients(loss)
    return train.apply_gradients(grads_and_vars)
