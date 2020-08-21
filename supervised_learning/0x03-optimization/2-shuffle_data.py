#!/usr/bin/env python3
"""Shuffle data"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way"""
    m, nx = Y.shape
    shuffled = np.random.permutation(m)
    X_shuffled = X[shuffled]
    Y_shuffled = Y[shuffled]
    return X_shuffled, Y_shuffled
