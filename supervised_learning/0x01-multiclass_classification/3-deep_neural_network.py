#!/usr/bin/env python3
"""Deep Neural Network class"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Class that defines a neural network with one hidden performing
    binary classification
    """

    @staticmethod
    def he_et_al(nx, layers):
        """Calculates weights using he et al method"""
        weights = dict()
        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')
            prev_layer = layers[i - 1] if i > 0 else nx
            w_part1 = np.random.randn(layers[i], prev_layer)
            w_part2 = np.sqrt(2 / prev_layer)
            weights.update({
                'b' + str(i + 1): np.zeros((layers[i], 1)),
                'W' + str(i + 1): w_part1 * w_part2
            })
        return weights

    def __init__(self, nx, layers):
        """Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = dict()
        self.__weights = self.he_et_al(nx, layers)

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @staticmethod
    def plot_training_cost(list_iterations, list_cost, graph):
        """Plots graph"""
        if graph:
            plt.plot(list_iterations, list_cost)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training cost')
            plt.show()

    @staticmethod
    def print_verbose_for_step(iteration, cost, verbose, step, list_cost):
        """Prints cost for each iteration"""
        if verbose and iteration % step == 0:
            print('Cost after ' + str(iteration) + ' iterations: ' + str(cost))
        list_cost.append(cost)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, "rb") as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError as e:
            return None

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if '.pkl' not in filename:
            filename += '.pkl'
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def forward_prop(self, X):
        """Calculates the forward propagation of the deep neural network

        Args:
            X: input data

        Returns:
            Output of the neural network and the cache
        """
        self.cache.update({'A0': X})
        for i in range(self.L):
            A = self.cache.get('A' + str(i))
            biases = self.weights.get('b' + str(i + 1))
            weights = self.weights.get('W' + str(i + 1))
            Z = np.matmul(weights, A) + biases
            if i + 1 == self.L:
                t = np.exp(Z)
                a = t / np.sum(t, axis=0, keepdims=True)
                self.cache.update({'A' + str(i + 1): a})
            else:
                self.cache.update({'A' + str(i + 1): 1 / (1 + np.exp(-Z))})

        return self.cache.get('A' + str(i + 1)), self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression

        Args:
            Y: contains the correct labels for the input data
            A: containing the activated output of the neuron for each example

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(np.multiply(Y, np.log(A)))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data

        Returns:
            The neuron's prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        return np.where(A <= 0.5, 0, 1), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the deep neural network

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
            cache: all intermediary values of the network
            alpha: learning rate
        """
        n_layers = range(self.L, 0, -1)
        m = Y.shape[1]
        dZ_prev = 0
        weights = self.weights.copy()

        for i in n_layers:
            A = cache.get('A' + str(i))
            A_prev = cache.get('A' + str(i - 1))
            weights_i = weights.get('W' + str(i))
            weights_n = weights.get('W' + str(i + 1))
            biases = weights.get('b' + str(i))
            if i == self.L:
                dZ = A - Y
            else:
                dZ = np.matmul(weights_n.T, dZ_prev) * (A * (1 - A))
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.__weights['W' + str(i)] = weights_i - (dW * alpha)
            self.__weights['b' + str(i)] = biases - (db * alpha)
            dZ_prev = dZ

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network

        Args:
            X: contains the input data
            Y: contains the correct labels for the input data
            iterations: number of iterations to train over
            alpha: learning rate

        Returns:
            The evaluation of the training data after iterations of training
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
            if i < iterations:
                self.gradient_descent(Y, self.cache, alpha)

        self.plot_training_cost(list_iterations, list_cost, graph)
        return self.evaluate(X, Y)
