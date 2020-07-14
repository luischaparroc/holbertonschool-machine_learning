#!/usr/bin/env python3
""" Returns the transpose of a 2D matrix """


def get_length(rows):
    """Recursive function used to calculate the length

    Returns:
        List with the length
    """
    if type(rows) is list or type(rows) is tuple:
        return [len(rows), *get_length(rows[0])]
    return []


def matrix_shape(matrix):
    """Calculates the shape of a matrix

    Returns:
        List with the shape of the matrix
    """
    return [*get_length(matrix)]
