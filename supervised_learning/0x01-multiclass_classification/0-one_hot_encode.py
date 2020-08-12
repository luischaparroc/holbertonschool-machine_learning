#!/usr/bin/env python3
"""One-hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix"""
    if not isinstance(Y, np.ndarray):
        return None
    if type(classes) is not int:
        return None
    if len(Y) == 0:
        return None
    if classes <= np.amax(Y):
        return None

    m, *_ = Y.shape
    y_one_hot = np.zeros((classes, m))

    for num in set(Y):
        indexes = np.where(Y == num)
        y_one_hot[num, indexes] = 1

    return y_one_hot
