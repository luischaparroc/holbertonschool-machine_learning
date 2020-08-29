#!/usr/bin/env python3
"""Dropout Forward Propagation"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = dict()
    cache.update({'A0': X})
    for i in range(L):
        A = cache.get('A' + str(i))
        b = weights.get('b' + str(i + 1))
        w = weights.get('W' + str(i + 1))
        Z = np.matmul(w, A) + b
        if i + 1 == L:
            t = np.exp(Z)
            a = t / np.sum(t, axis=0, keepdims=True)
        else:
            dropout = np.random.rand(Z.shape[0], Z.shape[1])
            dropout = np.where(dropout < keep_prob, 1, 0)
            cache.update({'D' + str(i + 1): dropout})
            a = np.tanh(Z) * dropout / keep_prob
        cache.update({'A' + str(i + 1): a})

    return cache
