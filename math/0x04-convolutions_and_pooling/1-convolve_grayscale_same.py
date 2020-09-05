#!/usr/bin/env python3
"""Convolve grayscale same"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    if kh % 2 == 0:
        ph = int(kh / 2)
    else:
        ph = int((kh - 1) / 2)

    if kw % 2 == 0:
        pw = int(kw / 2)
    else:
        pw = int((kw - 1) / 2)

    input_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    output = np.zeros((m, h, w))
    rng_im = np.arange(m)

    for i_h in range(h):
        for i_w in range(w):
            filtered = input_padded[rng_im, i_h:kh+i_h, i_w:kw+i_w]
            output[rng_im, i_h, i_w] = np.sum(filtered * kernel, axis=(1, 2))
    return output
