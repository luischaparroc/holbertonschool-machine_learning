#!/usr/bin/env python3
"""Neuron class"""
import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """W getter"""
        return self.__W

    @property
    def b(self):
        """b getter"""
        return self.__b

    @property
    def A(self):
        """A getter"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        Args:
            X: input data

        Returns:
            Activation function - calculated with sigmoid function
        """
        A_prev = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-A_prev))
        return self.__A
