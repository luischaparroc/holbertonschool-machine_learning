#!/usr/bin/env python3
""" Performs matrix multiplication """


def mat_mul(mat1, mat2):
    """Multiplicates two 2D matrices

    Returns:
        Matrix multiplication
    """
    if len(mat1[0]) != len(mat2):
        return None

    r1 = range(len(mat1))  # range of matrix 1
    r2 = range(len(mat2[0]))  # range of matrix 2
    rax = range(len(mat1[0]))  # range of common axis

    r = [[sum([mat1[i][k] * mat2[k][j] for k in rax]) for j in r2] for i in r1]
    return r
