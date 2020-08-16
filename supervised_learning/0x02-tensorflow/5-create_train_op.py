#!/usr/bin/env python3
"""Create train operation module"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Function that creates the training operation
    for the network
    """
    train = tf.train.GradientDescentOptimizer(alpha)
    grads_and_vars = train.compute_gradients(loss)
    return train.apply_gradients(grads_and_vars)
