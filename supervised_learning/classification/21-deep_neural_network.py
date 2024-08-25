#!/usr/bin/env python3
"""
defines the DeepNeuralNetwork class that
performs binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network.
        Parameters:
        -----------
        nx : int
            Number of input features
        layers : list
            A list containing the number of neurons in each layer
        Raises:
        -------
        TypeError:
            If nx is not an integer or if layers is not
            a list of positive integers.
        ValueError:
            If nx is less than 1 or if layers is an empty list.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Initialize the number of layers, cache, and weights dictionary
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize layers and validate
        for layer_index in range(self.__L):
            layer_size = layers[layer_index]
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            prev_layer_size = (nx if layer_index == 0 else
                               layers[layer_index - 1])

            # He initialization: random weights scaled by sqrt
            # (2 / number of inputs)
            self.__weights['W' + str(layer_index + 1)] = (
                np.random.randn(layer_size, prev_layer_size)
                * np.sqrt(2 / prev_layer_size)
            )
            # column vector with a size equal to the number
            # of neurons in this layer
            self.__weights['b' + str(layer_index + 1)] = (
              np.zeros((layer_size, 1))
              )

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache (intermediary values)."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights (weights and biases)."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (nx, m) where nx is the number
            of input features and m is the number of examples.
        Returns:
        --------
        A : numpy.ndarray
            The output of the neural network after forward propagation.
        cache : dict
            Dictionary containing all the intermediary
            activations in the network.
        """
        self.__cache['A0'] = X

        # Perform forward propagation through each layer
        for layer_index in range(self.__L):
            W = self.__weights['W' + str(layer_index + 1)]
            b = self.__weights['b' + str(layer_index + 1)]
            A_prev = self.__cache['A' + str(layer_index)]

            Z = np.matmul(W, A_prev) + b  # Linear combination
            A = 1 / (1 + np.exp(-Z))      # Sigmoid activation
            # Store the activated output in the cache
            self.__cache['A' + str(layer_index + 1)] = A
        # The final layer's output is the output of the neural network
        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Parameters:
        -----------
        Y : numpy.ndarray
            Shape (1, m) that contains the correct labels 4 the input data.
        A : numpy.ndarray
            Shape (1, m) containing the activated output
            of the neuron 4 each example.
        Returns:
        --------
        cost : float
            The cost of the model.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.
        Parameters:
        -----------
        X : numpy.ndarray
            Shape (nx, m) that contains the input data.
        Y : numpy.ndarray
            Shape (1, m) that contains the correct labels 4 the input data.
        Returns:
        --------
        prediction : numpy.ndarray
            Shape (1, m) containing the predicted labels 4 each example.
        cost : float
            The cost of the network.
        """
        # Perform forward propagation
        A, _ = self.forward_prop(X)
        # Convert the output probabilities to binary predictions (0 or 1)
        prediction = np.where(A >= 0.5, 1, 0)
        # Calculate the cost using the predicted activations
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.

        Parameters:
        -----------
        Y : numpy.ndarray
            Shape (1, m) that contains the correct labels 4 the input data.
        cache : dict
            Dictionary containing all the intermediary values of the network.
        alpha : float
            The learning rate.
        Updates:
        --------
        Updates the private attribute __weights.
        """
        m = Y.shape[1]
        A_last = cache['A' + str(self.__L)]
        delta_Z = A_last - Y

        for layer_index in reversed(range(1, self.__L + 1)):
            A_prev = cache['A' + str(layer_index - 1)]
            W = self.__weights['W' + str(layer_index)]

            dW = np.matmul(delta_Z, A_prev.T) / m
            db = np.sum(delta_Z, axis=1, keepdims=True) / m
            dA_prev = np.matmul(W.T, delta_Z)
            # Update weights and biases
            self.__weights['W' + str(layer_index)] -= alpha * dW
            self.__weights['b' + str(layer_index)] -= alpha * db
            # If not the first layer, calculate delta_Z 4 the next layer
            if layer_index > 1:
                delta_Z = dA_prev * A_prev * (1 - A_prev)
