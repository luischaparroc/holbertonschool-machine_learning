#!/usr/bin/env python3
""" Slices a matrix along a specific axes """


def np_slice(matrix, axes={}):
    """Slices a passed matrix with some specified axis

    Returns:
        New sliced matrix
    """
    h_axis = max(axes, key=int) + 1  # highest axis
    slice_obj = tuple([slice(*axes.get(i) or (None,)) for i in range(h_axis)])
    return matrix[slice_obj]
