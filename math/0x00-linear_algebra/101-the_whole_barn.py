#!/usr/bin/env python3
""" Adds two matrices element-wise """


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


def add_rows(row1, row2):
    """Add numbers on the same row level recursively

    Returns:
        List or matrix with the addition result
    """
    if type(row1[0]) is list:
        return [add_rows(row1[i], row2[i]) for i in range(len(row1))]
    return [row1[i] + row2[i] for i in range(len(row1))]


def add_matrices(mat1, mat2):
    """Adds two matrices 2D

    Returns:
        New matrix with addition
    """

    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    return add_rows(mat1, mat2)
