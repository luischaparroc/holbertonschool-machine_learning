#!/usr/bin/env python3
"""Module with Poisson class"""


class Poisson:
    """Class that represents a Poisson distribution"""

    EULER_NUMBER = 2.7182818285

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

    @staticmethod
    def factorial(n):
        """Calculates factorial of given number

        Args:
            n: input number

        Returns:
            response of n factorial
        """
        if n == 0:
            return 1
        return n * Poisson.factorial(n - 1)

    def pmf(self, k):
        """Calculates Probability Mass Function (PMF)

        Args:
            k: number of successes

        Returns:
            PMF of k or 0 if k is out of range.
        """
        k = int(k)

        if k < 0:
            return 0

        num = (self.lambtha ** k) * (self.EULER_NUMBER ** -self.lambtha)
        den = self.factorial(k)

        return num/den

    def cdf(self, k):
        """Calculates Cumulative Distribution Function (CDF)

        Args:
            k: number of successes

        Returns:
            CDF of k or 0 if k is out of range.
        """
        k = int(k)

        if k < 0:
            return 0

        summation = 0
        for i in range(k + 1):
            summation += (self.lambtha ** i) / (self.factorial(i))

        return (self.EULER_NUMBER ** -self.lambtha) * summation
