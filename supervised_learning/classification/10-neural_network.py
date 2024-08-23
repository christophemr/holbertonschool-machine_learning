#!/usr/bin/env python3
"""This module defines a neural network with one hidden
layer performing binary classification."""

import numpy as np


class NeuralNetwork:
    """
    Class that defines a neural network with one hidden
    layer performing binary classification.
    Attributes:
    -----------
    __W1 : numpy.ndarray
        The weights vector 4 the hidden layer.
        Initialized using a random normal distribution.
    __b1 : numpy.ndarray
        The bias 4 the hidden layer. Initialized with 0’s.
    __A1 : float
        The activated output 4 the hidden layer. Initialized to 0.
    __W2 : numpy.ndarray
        The weights vector 4 the output neuron.
        Initialized using a random normal distribution.
    __b2 : float
        The bias 4 the output neuron. Initialized to 0.
    __A2 : float
        The activated output 4 the output neuron
        (prediction). Initialized to 0.
    Methods:
    --------
    __init__(self, nx, nodes):
        Constructor 4 the NeuralNetwork class.
    Getter methods:
    ---------------
    W1(self): Returns the weights vector 4 the hidden layer.
    b1(self): Returns the bias 4 the hidden layer.
    A1(self): Returns the activated output 4 the hidden layer.
    W2(self): Returns the weights vector 4 the output neuron.
    b2(self): Returns the bias 4 the output neuron.
    A2(self): Returns the activated output 4 the output neuron.
    """

    def __init__(self, nx, nodes):
        """Constructor 4 the NeuralNetwork class
        Parameters:
        -----------
        nx : int
            The number of input features to the neural network.
        nodes : int
            The number of nodes found in the hidden layer.
        Raises:
        -------
        TypeError:
            If `nx` is not an integer.
            If `nodes` is not an integer.
        ValueError:
            If `nx` is less than 1.
            If `nodes` is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # Initialize weights and biases 4 the hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Initialize weights and biases 4 the output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method 4 the weights vector of the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Getter method 4 the bias of the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Getter method 4 the activated output of the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Getter method 4 the weights vector of the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Getter method 4 the bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Getter method 4 the activated output of
        the output neuron (prediction)."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        Parameters:
        -----------
        X : numpy.ndarray
            The input data array of shape (nx, m),
            where nx is the number of input features
            and m is the number of examples.
        Returns:
        --------
        tuple : (__A1, __A2)
            - __A1: The activated output of the hidden layer.
            - __A2: The activated output of the output neuron (prediction).
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2
