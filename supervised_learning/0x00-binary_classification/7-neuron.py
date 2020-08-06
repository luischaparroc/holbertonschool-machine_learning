#!/usr/bin/env python3
"""Neuron class"""
import numpy as np
import matplotlib.pyplot as plt


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

    @staticmethod
    def plot_training_cost(list_iterations, list_cost, graph):
        if graph:
            plt.plot(list_iterations, list_cost)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training cost')
            plt.show()

    @staticmethod
    def print_verbose_for_step(iteration, cost, verbose, step, list_cost):
        if verbose and iteration % step == 0:
            print(f'Cost after {iteration} iterations: {cost}')
        list_cost.append(cost)

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
        """Evaluates the neuron's predictions

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data

        Returns:
            The neuron's prediction and the cost of the network
        """
        self.forward_prop(X)
        return np.where(self.A <= 0.5, 0, 1), self.cost(Y, self.A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
            A: containing the activated output of the neuron for each example
            alpha: learning rate
        """
        m = Y.shape[1]
        d_ay = A - Y
        gradient = np.matmul(d_ay, X.T) / m
        db = np.sum(d_ay) / m
        self.__W -= gradient * alpha
        self.__b -= db * alpha

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains a neuron

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
            iterations: number of iterations to train over
            alpha: learning rate
            verbose: boolean to print information about the training
            graph: boolean to graph training once finishes
            step: step to print learning information
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        if verbose and graph:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if not 0 <= step <= iterations:
                raise ValueError('step must be positive and <= iterations')

        list_cost = list()
        list_iterations = [*list(range(iterations)), iterations]

        for i in list_iterations:
            A, cost = self.evaluate(X, Y)
            self.print_verbose_for_step(i, cost, verbose, step, list_cost)
            self.gradient_descent(X, Y, self.A, alpha)

        self.plot_training_cost(list_iterations, list_cost, graph)
        return A, cost
