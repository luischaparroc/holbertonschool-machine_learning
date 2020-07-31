#!/usr/bin/env python3
"""Module with Binomial class"""


class Binomial:
    """Class that represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor"""
        if data is not None:
            n, p = self.calculate_n_p(data)
        self.n = n
        self.p = p

    @property
    def n(self):
        """n getter"""
        return self.__n

    @n.setter
    def n(self, n):
        """n setter"""
        if n <= 0:
            raise ValueError('n must be a positive value')
        self.__n = int(n)

    @property
    def p(self):
        """p getter"""
        return self.__p

    @p.setter
    def p(self, p):
        """p setter"""
        if not 0 < p < 1:
            raise ValueError('p must be greater than 0 and less than 1')
        self.__p = float(p)

    @staticmethod
    def factorial(n):
        """Calculates factorial of given number

        Args:
            n: input number

        Returns:
            response of n factorial
        """
        factorial_n = 1
        for i in range(1, n + 1):
            factorial_n *= i
        return factorial_n

    @classmethod
    def calculate_n_p(cls, data):
        """Calculates n Bernoulli trials and probability"""
        if not isinstance(data, list):
            raise TypeError('data must be a list')
        if len(data) < 2:
            raise ValueError('data must contain multiple values')

        len_data = len(data)
        mean = sum(data)/len_data
        variance = sum([(number - mean) ** 2 for number in data])/len_data
        p = 1 - (variance/mean)
        n = int(round(mean/p))
        p = (mean/n)
        return n, p

    def pmf(self, k):
        """Calculates Probability Mass Function (PMF)
        Args:
            k: number of successes
        Returns:
            PMF of k or 0 if k is out of range.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        binomial_coefficient = self.get_bcf(k)
        q = 1 - self.p
        return binomial_coefficient * ((self.p ** k) * (q ** (self.n - k)))

    def cdf(self, k):
        """Calculates Cumulative Distribution Function (CDF)

        Args:
            k: number of successes

        Returns:
            CDF of k or 0 if k is out of range.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        return sum([self.cdf_equation(i) for i in range(k + 1)])

    def get_bcf(self, k):
        """Calculates binomial coefficient with a given number"""
        n_factorial = self.factorial(self.n)
        k_factorial = self.factorial(k)
        n_k_factorial = self.factorial(self.n - k)
        binomial_coefficient = n_factorial/(n_k_factorial * k_factorial)
        return binomial_coefficient

    def cdf_equation(self, i):
        """Calculates cdf for each iteration"""
        r = self.get_bcf(i) * ((self.p ** i) * ((1 - self.p) ** (self.n - i)))
        return r
