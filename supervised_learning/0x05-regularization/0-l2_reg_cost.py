#!/usr/bin/env python3
"""l2 reg cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates L2 Regularization cost"""
    range_l = range(1, L + 1)
    reg = np.sum([np.linalg.norm(weights['W{}'.format(l)]) for l in range_l])
    l2_reg = cost + (lambtha / (2 * m)) * reg
    return l2_reg
