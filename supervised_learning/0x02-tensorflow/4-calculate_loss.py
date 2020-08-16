#!/usr/bin/env python3
"""Calculate loss module"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Function that calculates the softmax cross-entropy loss
    of a prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
