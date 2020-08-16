#!/usr/bin/env python3
"""Calculate accuracy module"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy
    of a single prediction
    """
    prediction = tf.equal(
        tf.argmax(y_pred, 1),
        tf.argmax(y, 1)
    )
    return tf.reduce_mean(tf.cast(prediction, tf.float32))
