#!/usr/bin/env python3
"""Update variables Adam"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2,
                          epsilon, var, grad, v, s, t):
    """Updates a variable in place using
    the Adam optimization algorithm
    """
    v = (beta1 * v) + (1 - beta1) * grad
    v_c = v / (1 - (beta1 ** t))

    s = (beta2 * s) + (1 - beta2) * (grad ** 2)
    s_c = s / (1 - (beta2 ** t))

    var -= (alpha * (v_c / (np.sqrt(s_c) + epsilon)))

    return var, v, s
