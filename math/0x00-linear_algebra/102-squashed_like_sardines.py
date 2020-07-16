#!/usr/bin/env python3
""" Concatenates two matrices along a specific axis """


def get_length(rows):
    """Recursive function used to calculate the length

    Returns:
        List with the length
    """
    if rows and (type(rows) is list or type(rows) is tuple):
        return [len(rows), *get_length(rows[0])]
    return []


def matrix_shape(matrix):
    """Calculates the shape of a matrix

    Returns:
        List with the shape of the matrix
    """
    return [*get_length(matrix)]


def concatenate_rows(row1, row2, axis):
    """Concatenates rows recursively

    Returns:
        List or matrix concatenated
    """
    if axis == 0:
        return row1 + row2

    range_row = range(len(row1))
    return [concatenate_rows(row1[i], row2[i], axis - 1) for i in range_row]


def cat_matrices(mat1, mat2, axis=0):
    """ Concatenates two matrices along a specific axis

    Returns:
        New concatenated matrix
    """

    shape_matrix1 = matrix_shape(mat1)
    shape_matrix2 = matrix_shape(mat2)

    if shape_matrix1[axis + 1:] != shape_matrix2[axis + 1:]:
        return None

    return concatenate_rows(mat1, mat2, axis)
