#!/usr/bin/env python3
"""this module defines a deep neural network
performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network 4 binary classification.
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

        #  Initialize the number of layers, cache, and weights dictionary
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize layers and validate them in a single loop
        for layer_index in range(self.L):
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
        """Getter 4 L (number of layers)."""
        return self.__L

    @property
    def cache(self):
        """Getter 4 cache (intermediary values)."""
        return self.__cache

    @property
    def weights(self):
        """Getter 4 weights (weights and biases)."""
        return self.__weights
