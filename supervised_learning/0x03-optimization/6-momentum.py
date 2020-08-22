#!/usr/bin/env python3
"""Create momentum op"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a neural network,
    using gradient descent with momentum optimization algorithm
    """
    train = tf.train.MomentumOptimizer(alpha, beta1)
    grads_and_vars = train.compute_gradients(loss)
    return train.apply_gradients(grads_and_vars)
