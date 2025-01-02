#!/usr/bin/env python3
"""
Defines the function bi_rnn for forward propagation in a bidirectional RNN.
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Parameters:
        bi_cell (BidirectionalCell): An instance of BidirectionalCell.
        X (numpy.ndarray): Input data of shape (t, m, i).
            t: Maximum number of time steps.
            m: Batch size.
            i: Dimensionality of the data.
        h_0 (numpy.ndarray): Initial hidden state in the forward direction
        h_t (numpy.ndarray): Initial hidden state in the backward direction

    Returns:
        H (numpy.ndarray): Concatenated hidden states, shape (t, m, 2 * h).
        Y (numpy.ndarray): Outputs, shape (t, m, o).
    """
    t, m, i = X.shape
    _, h = h_0.shape

    # Forward pass
    H_forward = np.zeros((t, m, h))
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        H_forward[step] = h_prev

    # Backward pass
    H_backward = np.zeros((t, m, h))
    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        H_backward[step] = h_next

    # Concatenate forward and backward hidden states
    H = np.concatenate((H_forward, H_backward), axis=2)

    # Calculate outputs
    Y = bi_cell.output(H)

    return H, Y
