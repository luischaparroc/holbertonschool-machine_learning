#!/usr/bin/env python3
""" Adds two matrices element-wise """


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


def add_matrices2D(mat1, mat2):
    """Adds two matrices 2D

    Returns:
        New list with addition
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    response = []

    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        response.append(row)

    return response
