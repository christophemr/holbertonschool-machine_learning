#!/usr/bin/env python3
"""
Module for updating weights and biases of a neural network
using gradient descent with L2 regularization.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization.

    Parameters:
    Y (numpy.ndarray): One-hot matrix of shape (classes, m) containing the
        correct labels for the data.
    weights (dict): Dictionary of the weights and biases of the neural network.
    cache (dict): Dictionary of the outputs of each layer of
    the neural network.
    alpha (float): Learning rate.
    lambtha (float): L2 regularization parameter.
    L (int): Number of layers in the neural network.

    Returns:
    None: The weights and biases are updated in place.
    """
    m = Y.shape[1]  # Number of data points
    A_L = cache['A' + str(L)]  # Output from the last layer

    # Compute the gradient for the output layer (softmax layer)
    dZ = A_L - Y  # dZ for the last layer (softmax)
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]  # Activation from the previous layer
        # Gradient for weights with L2 regularization
        dW = (np.dot(dZ, A_prev.T) + lambtha * weights['W' + str(i)]) / m
        # Gradient for biases
        db = np.sum(dZ, axis=1, keepdims=True) / m
        # Update the weights and biases
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
        if i > 1:
            # Compute dA (activation gradient) for the next layer
            dA_prev = np.dot(weights['W' + str(i)].T, dZ)
            # Apply derivative of tanh activation function
            dZ = dA_prev * (1 - np.square(cache['A' + str(i - 1)]))
