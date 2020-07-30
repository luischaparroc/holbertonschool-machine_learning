#!/usr/bin/env python3
"""Module with Exponential class"""


class Normal:
    """Class that represents a normal distribution"""

    EULER_NUMBER = 2.7182818285
    PI = 3.1415926536

    @staticmethod
    def get_stddev(data, mean):
        """Calculates Standard Deviation with a given data and mean"""
        summation = 0
        for number in data:
            summation += (number - mean) ** 2
        return (summation/len(data)) ** (1/2)

    def __init__(self, data=None, mean=0, stddev=1.):
        """Class constructor"""
        if data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data)/len(data)
            self.stddev = self.get_stddev(data, self.mean)
        else:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """Calculates z_score

        Args:
            x: x-value

        Returns:
            z-score
        """
        return (x - self.mean)/self.stddev

    def x_value(self, z):
        """Calculates x_value

        Args:
            z: z-score

        Returns:
            x-value
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates Probability Density Function (PDF)

        Args:
            x: x-value

        Returns:
            PDF
        """
        exp_component = (-1/2) * (((x - self.mean)/self.stddev) ** 2)
        euler_exp = self.EULER_NUMBER ** exp_component
        cdf = euler_exp/(self.stddev * ((2 * self.PI) ** (1/2)))
        return cdf
