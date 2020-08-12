#!/usr/bin/env python3
"""One-hot Decode"""
import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray):
        return None
    if len(one_hot.shape) != 2:
        return None
    if np.any((one_hot != 0) & (one_hot != 1)):
        return None

    classes, m = one_hot.shape
    y_decoded = np.zeros(m, dtype=int)

    for i, arrays in enumerate(one_hot):
        indexes = np.where(arrays == 1)
        y_decoded[indexes] = i

    return y_decoded
