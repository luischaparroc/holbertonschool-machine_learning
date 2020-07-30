#!/usr/bin/env python3
"""Module with class Poisson"""


class Poisson:
    """Class that represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Constructor Poisson class

        Args:
            data: list of the data to be used to estimate the distribution
            lambtha: expected number of occurrences in a given time frame
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data)/len(data)
        else:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
