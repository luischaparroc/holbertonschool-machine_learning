#!/usr/bin/env python3
"""Convolve grayscale valid"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    oh = h - kh + 1
    ow = w - kw + 1
    output = np.zeros((m, oh, ow))
    rng_im = np.arange(m)

    for i_oh in range(oh):
        for i_ow in range(ow):
            filtered = images[rng_im, i_oh:kh+i_oh, i_ow:kw+i_ow]
            output[rng_im, i_oh, i_ow] = np.sum(filtered * kernel, axis=(1, 2))
    return output
