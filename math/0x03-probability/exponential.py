#!/usr/bin/env python3
"""Module with Exponential class"""


class Exponential:
    """Class that represents an exponential distribution"""

    EULER_NUMBER = 2.7182818285

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

    def pdf(self, x):
        """Calculates Probability Density Function (PDF)

        Args:
            x: time period

        Returns:
            PDF of x or 0 if x is out of range.
        """
        if x < 0:
            return 0

        return self.lambtha * (self.EULER_NUMBER ** (-self.lambtha * x))

    def cdf(self, x):
        """Calculates Cumulative Distribution Function (CDF)

        Args:
            x: time period

        Returns:
            PDF of x or 0 if x is out of range.
        """
        if x < 0:
            return 0

        return 1 - (self.EULER_NUMBER ** (-self.lambtha * x))
