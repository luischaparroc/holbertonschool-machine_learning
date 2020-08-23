#!/usr/bin/env python3
"""Create batch norm"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network
    using batch normalization
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    Z_norm = (Z - mean) / (np.sqrt(var + epsilon))
    Z_n = gamma * Z_norm + beta
    return Z_n
