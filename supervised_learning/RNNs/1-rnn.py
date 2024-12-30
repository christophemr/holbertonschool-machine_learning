#!/usr/bin/env python3
"""
Defines function that performs forward propagation for simple RNN
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Parameters:
    - rnn_cell: instance of RNNCell that will be used for forward propagation
    - X: numpy.ndarray of shape (t, m, i), data to be used
         t: maximum number of time steps
         m: batch size
         i: dimensionality of the data
    - h_0: numpy.ndarray of shape (m, h), initial hidden state
         m: batch size
         h: dimensionality of the hidden state

    Returns:
    - H: numpy.ndarray containing all of the hidden states
    - Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape

    # Initialize the hidden states (t+1 steps including initial state)
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Initialize the output array
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    # Perform forward propagation through all time steps
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
