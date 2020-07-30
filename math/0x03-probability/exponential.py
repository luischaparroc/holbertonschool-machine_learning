#!/usr/bin/env python3
"""Module with Exponential class"""


class Exponential:
    """Class that represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Class constructor"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1/(sum(data)/len(data))
        else:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
