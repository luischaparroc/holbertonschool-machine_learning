#!/usr/bin/env python3
"""Concatenates two matrices"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis

    Returns:
        matrix concatenated as a numpy.ndarray
    """
    return np.concatenate((mat1, mat2), axis=axis)
