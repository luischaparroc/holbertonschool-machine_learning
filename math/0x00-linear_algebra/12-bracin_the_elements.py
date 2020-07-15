#!/usr/bin/env python3
""" Performs element-wise operations"""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition, subtraction, multiplication
    and division

    Returns:
        tuple containing the element-wise operations
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
