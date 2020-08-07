#!/usr/bin/env python3
"""Neural Network class"""
import numpy as np


class NeuralNetwork:
    """Class that defines a neural network with one hidden performing
    binary classification
    """

    def __init__(self, nx, nodes):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """W1 getter"""
        return self.__W1

    @property
    def b1(self):
        """b1 getter"""
        return self.__b1

    @property
    def A1(self):
        """A1 getter"""
        return self.__A1

    @property
    def W2(self):
        """W2 getter"""
        return self.__W2

    @property
    def b2(self):
        """b2 getter"""
        return self.__b2

    @property
    def A2(self):
        """A2 getter"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        Args:
            X: input data

        Returns:
            Activation functions (A1 & A2) - calculated with sigmoid function
        """
        A1_prev = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-A1_prev))

        A2_prev = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-A2_prev))

        return self.A1, self.A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: contains the correct labels for the input data
            A: containing the activated output of the neuron for each example

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                  np.multiply(1 - Y, np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data

        Returns:
            The neuron's prediction and the cost of the network
        """
        self.forward_prop(X)
        return np.where(self.A2 <= 0.5, 0, 1), self.cost(Y, self.A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network

         Args:
             X: contains the input data
             Y: contains the correct labels for the input data
             A1: the output of the hidden layer
             A2: predicted output
             alpha: learning rate
         """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.matmul(self.W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 -= dW1 * alpha
        self.__b1 -= db1 * alpha
        self.__W2 -= dW2 * alpha
        self.__b2 -= db2 * alpha
