#!/usr/bin/env python3
"""Normalization constants"""
import numpy as np


def normalization_constants(X):
    """Returns mean and standard deviation of input data"""
    X_ref = X.copy()
    m, nx = X_ref.shape
    mean = (1 / m) * X_ref.sum(axis=0)
    X_ref -= mean
    X_ref = X_ref ** 2
    var = (1 / m) * X_ref.sum(axis=0)
    dv = np.sqrt(var)
    return mean, dv
