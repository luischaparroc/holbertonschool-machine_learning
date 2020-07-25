#!/usr/bin/env python3
"""Function that calculates a summation"""


def summation_i_squared(n):
    """Function that returns the result of the summation

    Args:
        n: limit

    Returns:
        Summation response
    """
    if type(n) is not int or n < 1:
        return None
    return (n*(n + 1) * (2*n + 1))//6
