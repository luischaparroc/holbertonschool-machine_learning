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
