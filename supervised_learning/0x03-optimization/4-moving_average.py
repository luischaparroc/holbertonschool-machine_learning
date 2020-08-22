#!/usr/bin/env python3
"""Moving average"""


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set"""
    weighted_data = list()
    epsilon = 1 - beta
    V = 0

    for i, theta in enumerate(data, start=1):
        bias_correction = 1 - (beta ** i)
        V = (beta * V) + (epsilon * theta)
        weighted_data.append(V / bias_correction)

    return weighted_data
