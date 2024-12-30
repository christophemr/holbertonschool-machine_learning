#!/usr/bin/env python3
"""
Defines the class RNNCell that represents a cell of a simple RNN
"""

import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        Initializes the RNN cell with weights and biases.

        Parameters:
        - i: Dimensionality of the data
        - h: Dimensionality of the hidden state
        - o: Dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        Performs the softmax function.

        Parameters:
        - x: Numpy array for which to compute softmax

        Returns:
        - Softmax-transformed numpy array
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
        - h_prev: numpy.ndarray of shape (m, h), previous hidden state
        - x_t: numpy.ndarray of shape (m, i), input data for this time step

        Returns:
        - h_next: The next hidden state
        - y: The output of the cell
        """
        # Concatenate previous hidden state and input data
        concatenation = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state
        h_next = np.tanh(np.dot(concatenation, self.Wh) + self.bh)

        # Compute output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
