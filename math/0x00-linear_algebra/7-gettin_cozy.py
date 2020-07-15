#!/usr/bin/env python3
""" Concatenates two matrices along a specific axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices along a specific axis

    Returns:
        New concatenated matrix
    """
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return [row.copy() for row in [*mat1, *mat2]]
    elif axis == 1 and len(mat1) == len(mat2):
        return [mat1[i].copy() + mat2[i].copy() for i in range(len(mat1))]
