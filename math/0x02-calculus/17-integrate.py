#!/usr/bin/env python3
"""Function that calculates the integral of a polynomial"""


def handle_whole_numbers(num):
    """Checks if a passed number is whole number to return an integer"""
    return int(num) if int(num) == num else num


def poly_integral(poly, C=0):
    """Function that calculated the integral of a polynomial

    Args:
        poly: list of coefficients
        C: integration constant

    Returns:
        response: new list of coefficients representing
                  the integral of the polynomial
    """
    if not poly or type(poly) is not list or type(C) is not int:
        return None

    if sum(poly):
        res = [handle_whole_numbers(c/(i + 1)) for i, c in enumerate(poly)]
    else:
        res = []

    res.insert(0, C)

    return res
