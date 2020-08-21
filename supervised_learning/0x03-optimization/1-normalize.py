#!/usr/bin/env python3
"""Normalization constants"""
import numpy as np


def normalize(X, m, s):
    """Returns mean and standard deviation of input data"""
    X -= m
    X /= s
    return X
