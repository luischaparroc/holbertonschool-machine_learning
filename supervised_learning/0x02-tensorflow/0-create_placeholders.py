#!/usr/bin/env python3
"""Create placeholders module"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Function that creates and returns placeholders
    for a neural network
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
