#!/usr/bin/env python3
"""Function that calculates a summation"""


def calculate_summation(i, n):
    """Function that calculates summation setting the series recursively

    Args:
        i: index (dummy variable)
        n: limit

    Returns:
        Summation
    """
    if i > n:
        return 0
    return (i ** 2) + calculate_summation(i + 1, n)


def summation_i_squared(n):
    """Function that returns the result of the summation

    Args:
        n: limit

    Returns:
        Summation response
    """
    if type(n) is not int or n < 1:
        return None
    return calculate_summation(1, n)
