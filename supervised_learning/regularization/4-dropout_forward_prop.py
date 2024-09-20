#!/usr/bin/env python3
"""
Module for forward propagation with dropout regularization.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Parameters:
    X (numpy.ndarray): Input data of shape (nx, m) where nx is the number
                       of input features and m is the number of examples.
    weights (dict): Dictionary containing the weights and biases.
    L (int): Number of layers in the network.
    keep_prob (float): Probability that a node will be kept during dropout.
    Returns:
    dict: Dictionary containing the outputs of each layer and dropout masks.
    """
    cache = {"A0": X}  # Cache to store intermediate outputs
    for i in range(1, L + 1):
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]
        A_prev = cache["A" + str(i - 1)]
        # Linear step Z = W.A_prev + b
        Z = np.dot(W, A_prev) + b
        # Apply activation functions
        if i == L:
            # Softmax for the last layer
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            # tanh for hidden layers
            A = np.tanh(Z)
            # Dropout for hidden layers
            D = (np.random.rand(A.shape[0],
                                A.shape[1]) < keep_prob).astype(int)
            A *= D  # Apply dropout mask
            A /= keep_prob  # Scale activations
            cache["D" + str(i)] = D  # Save the dropout mask
        # Store activation in cache
        cache["A" + str(i)] = A
    return cache
