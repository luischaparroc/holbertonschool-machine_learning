#!/usr/bin/env python3
"""Conv forward"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs a convolution on images with channels"""
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
    return activation(Z)
