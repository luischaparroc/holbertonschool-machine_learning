#!/usr/bin/env python3
"""Convolve grayscale padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1

    input_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    output = np.zeros((m, oh, ow))
    rng_im = np.arange(m)

    for i_oh in range(oh):
        for i_ow in range(ow):
            filtered = input_padded[rng_im, i_oh:kh+i_oh, i_ow:kw+i_ow]
            output[rng_im, i_oh, i_ow] = np.sum(filtered * kernel, axis=(1, 2))
    return output
