#!/usr/bin/env python3
"""this module defines a neural with one hidden layer
performing binary classification"""

import numpy as np


class NeuralNetwork:
    """
    Class that defines a neural network with one hidden
    layer performing binary classification.
    Attributes:
    -----------
    W1 : numpy.ndarray
        The weights vector for the hidden layer.
        Initialized using a random normal distribution.
    b1 : numpy.ndarray
        The bias for the hidden layer. Initialized with 0’s.
    A1 : float
        The activated output for the hidden layer. Initialized to 0.
    W2 : numpy.ndarray
        The weights vector for the output neuron.
        Initialized using a random normal distribution.
    b2 : float
        The bias for the output neuron. Initialized to 0.
    A2 : float
        The activated output for the output neuron (prediction).
        Initialized to 0.
    """

    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # Initialize weights and biases for the hidden layer
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        # Initialize weights and biases for the output layer
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
