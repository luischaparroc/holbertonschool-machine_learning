#!/usr/bin/env python3
"""Create Adam op"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network,, using
    the Adam optimization algorithm
    """
    train = tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )
    grads_and_vars = train.compute_gradients(loss)
    return train.apply_gradients(grads_and_vars)
