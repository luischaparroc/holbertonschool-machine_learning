#!/usr/bin/env python3
""" Returns the transpose of a 2D matrix """


def matrix_transpose(matrix):
    """Calcuates the transpose of a matrix

    Returns:
        List with the transpose
    """
    transpose = []
    for i in range(len(matrix[0])):
        new_row = []
        for j in range(len(matrix)):
            new_row.append(matrix[j][i])
        transpose.append(new_row)
    return transpose
