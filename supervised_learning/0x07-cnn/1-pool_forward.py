#!/usr/bin/env python3
"""Pool forward"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer of a neural network"""
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = int(((h - kh) / sh) + 1)
    ow = int(((w - kw) / sw) + 1)

    output = np.zeros((m, oh, ow, c))
    rng_im = np.arange(m)

    for i_oh in range(oh):
        for i_ow in range(ow):
            s_i_oh = i_oh * sh
            s_i_ow = i_ow * sw
            flt = A_prev[rng_im, s_i_oh: kh + s_i_oh, s_i_ow: kw + s_i_ow]
            if mode == 'max':
                output[rng_im, i_oh, i_ow] = flt.max(axis=(1, 2))
            elif mode == 'avg':
                output[rng_im, i_oh, i_ow] = np.mean(flt, axis=(1, 2))

    return output
