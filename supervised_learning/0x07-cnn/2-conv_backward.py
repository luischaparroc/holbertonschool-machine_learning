#!/usr/bin/env python3
"""Pool forward"""
import numpy as np


def forward_prop(self, X):
    """Calculates the forward propagation of the deep neural network

    Args:
        X: input data

    Returns:
        Output of the neural network and the cache
    """
    self.cache.update({'A0': X})
    for i in range(self.L):
        A = self.cache.get('A' + str(i))
        biases = self.weights.get('b' + str(i + 1))
        weights = self.weights.get('W' + str(i + 1))
        Z = np.matmul(weights, A) + biases
        self.cache.update({'A' + str(i + 1): 1 / (1 + np.exp(-Z))})

    return self.cache.get('A' + str(i + 1)), self.cache


def gradient_descent(self, Y, cache, alpha=0.05):
    """Calculates one pass of gradient descent on the deep neural network

    Args:
        X: contains the input data
        Y: contains the correct labels for the input data
        cache: all intermediary values of the network
        alpha: learning rate
    """
    n_layers = range(self.L, 0, -1)
    m = Y.shape[1]
    dZ_prev = 0
    weights = self.weights.copy()

    for i in n_layers:
        A = cache.get('A' + str(i))
        A_prev = cache.get('A' + str(i - 1))
        weights_i = weights.get('W' + str(i))
        weights_n = weights.get('W' + str(i + 1))
        biases = weights.get('b' + str(i))
        if i == self.L:
            dZ = A - Y
        else:
            dZ = np.matmul(weights_n.T, dZ_prev) * (A * (1 - A))
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        self.__weights['W' + str(i)] = weights_i - (dW * alpha)
        self.__weights['b' + str(i)] = biases - (db * alpha)
        dZ_prev = dZ


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs forward propagation over a pooling layer of a neural network"""
    m, h, w, c = A_prev.shape
    kh, kw, c, nc = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))

    oh = int(((h + 2 * ph - kh) / sh) + 1)
    ow = int(((w + 2 * pw - kw) / sw) + 1)

    input_pd = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        'constant'
    )

    output = np.zeros((m, oh, ow, nc))
    rng_im = np.arange(m)

    for k in range(nc):

        for i_oh in range(oh):
            for i_ow in range(ow):
                s_i_oh = i_oh * sh
                s_i_ow = i_ow * sw
                flt = input_pd[rng_im, s_i_oh:kh+s_i_oh, s_i_ow:kw+s_i_ow]
                kernel = W[:, :, :, k]
                output[rng_im, i_oh, i_ow, k] = np.sum(
                    flt * kernel, axis=(1, 2, 3)
                )
    Z = output + b
    return dA_prev, dW, db
