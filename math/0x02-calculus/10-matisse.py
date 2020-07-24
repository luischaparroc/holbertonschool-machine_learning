#!/usr/bin/env python3
"""Function that calculates a derivative"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polynomial

    Args:
        poly: list of coefficients representing a polynomial

    Returns:
        List of coefficients obtained by the derivative
    """
    if not poly or type(poly) is not list:
        return None

    response = []

    for order in range(1, len(poly)):
        response.append(order * poly[order])

    if not response:
        response.append(0)

    return response
