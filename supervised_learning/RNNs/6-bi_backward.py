#!/usr/bin/env python3
"""
Defines the BidirectionalCell class for a bidirectional RNN cell.
"""

import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN.

    Attributes:
        Whf (numpy.ndarray): Weights for hidden states (forward direction).
        Whb (numpy.ndarray): Weights for hidden states (backward direction).
        Wy (numpy.ndarray): Weights for outputs.
        bhf (numpy.ndarray): Biases for hidden states (forward direction).
        bhb (numpy.ndarray): Biases for hidden states (backward direction).
        by (numpy.ndarray): Biases for outputs.
    """

    def __init__(self, i, h, o):
        """
        Initializes the bidirectional RNN cell.

        Parameters:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden states.
            o (int): Dimensionality of the outputs.
        """
        self.Whf = np.random.normal(size=(i + h, h))  # Forward hidden weights
        self.bhf = np.zeros((1, h))                   # Forward hidden biases
        self.Whb = np.random.normal(size=(i + h, h))  # Backward hidden weights
        self.bhb = np.zeros((1, h))                   # Backward hidden biases
        self.Wy = np.random.normal(size=(2 * h, o))   # Output weights
        self.by = np.zeros((1, o))                    # Output biases

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.

        Parameters:
            h_prev (numpy.ndarray): Previous hidden state, shape (m, h).
            x_t (numpy.ndarray): Data input for the cell, shape (m, i).

        Returns:
            h_next (numpy.ndarray): The next hidden state.
        """
        concatenation = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenation, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step

        Parameters:
            h_next (numpy.ndarray): Next hidden state, shape (m, h).
            x_t (numpy.ndarray): Data input for the cell, shape (m, i).

        Returns:
            h_prev (numpy.ndarray): The previous hidden state.
        """
        concatenation = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(concatenation, self.Whb) + self.bhb)
        return h_prev
