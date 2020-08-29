#!/usr/bin/env python3
"""Early stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Early stopping"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count == patience, count
