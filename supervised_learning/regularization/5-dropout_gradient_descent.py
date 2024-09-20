#!/usr/bin/env python3
"""
Function that updates the weights of a neural network using
Dropout regularization with gradient descent
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.
    Parameters:
    Y (numpy.ndarray): One-hot array with shape (classes, m) containing
                       the correct labels for the data.
    weights (dict): Dictionary containing the weights and
    biases of the network.
    cache (dict): Dictionary containing the outputs and dropout masks of
                  each layer of the neural network.
    alpha (float): Learning rate.
    keep_prob (float): Probability that a node will be kept during dropout.
    L (int): Number of layers in the network.
    Returns:
    None: The weights and biases are updated in place.
    """
    m = Y.shape[1]
    A_L = cache["A" + str(L)]  # Output of the last layer (softmax)
    # Gradient of softmax output layer
    dZ = A_L - Y
    for i in range(L, 0, -1):
        A_prev = cache["A" + str(i - 1)]  # Activation from previous layer
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]
        # Compute dW and db
        dW = (np.dot(dZ, A_prev.T) / m)
        db = np.sum(dZ, axis=1, keepdims=True) / m
        if i > 1:
            # Gradient of the activation function (tanh)
            D = cache["D" + str(i - 1)]  # Dropout mask
            dA_prev = np.dot(W.T, dZ) * D  # Apply dropout mask
            dA_prev /= keep_prob  # Scale
            dZ = dA_prev * (1 - A_prev ** 2)  # Derivative of tanh
        # Update weights and biases
        weights["W" + str(i)] -= alpha * dW
        weights["b" + str(i)] -= alpha * db
