#!/usr/bin/env python3
"""l2 reg cost"""
import tensorflow as tf


def l2_reg_cost(cost):
    """Returns tensor with L2 regularization cost"""
    return cost + tf.losses.get_regularization_losses()
