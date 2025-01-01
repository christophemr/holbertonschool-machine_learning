#!/usr/bin/env python3
"""
Defines function that performs forward propagation for a deep RNN
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Parameters:
    - rnn_cells: list of RNNCell instances of length layers
    - X: numpy.ndarray of shape (t, m, i), data to be used
         t: number of time steps, m: batch size, i: input size
    - h_0: numpy.ndarray of shape (layers, m, h), initial hidden states

    Returns:
    - H: numpy.ndarray containing all of the hidden states
    - Y: numpy.ndarray containing all of the outputs
    """
    layers = len(rnn_cells)
    t, m, i = X.shape
    l, m, h = h_0.shape

    # Initialize hidden states
    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0

    outputs = []

    for step in range(t):
        x_t = X[step]
        for layer in range(layers):
            # Use previous layer's output or input X as input for
            # the current layer
            h_prev = H[step, layer]
            h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            H[step + 1, layer] = h_next
            x_t = h_next

        # Collect the output of the last layer
        outputs.append(y)

    # Stack all outputs to form a single array (t, m, o)
    Y = np.stack(outputs, axis=0)

    return H, Y
