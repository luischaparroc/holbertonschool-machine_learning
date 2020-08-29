#!/usr/bin/env python3
"""Dropout Forward Propagation"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Conducts forward propagation using Dropout"""
    n_layers = range(L, 0, -1)
    m = Y.shape[1]
    dZ_prev = 0
    w = weights.copy()

    for i in n_layers:
        A = cache['A{}'.format(i)]
        A_prev = cache['A{}'.format(i - 1)]
        weights_i = w.get('W{}'.format(i))
        weights_n = w.get('W{}'.format(i + 1))
        biases = w.get('b' + str(i))
        if i == L:
            dZ = A - Y
        else:
            dZ = np.matmul(weights_n.T, dZ_prev) * (1 - (A * A))
            dZ *= cache['D{}'.format(i)]
            dZ /= keep_prob
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        weights['W' + str(i)] = weights_i - (dW * alpha)
        weights['b' + str(i)] = biases - (db * alpha)
        dZ_prev = dZ
