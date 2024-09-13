#!/usr/bin/env python3
"""function that normalizes an unactivated output of a neural
network using batch normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network
    using batch normalization.

    Args:
        Z (numpy.ndarray): The matrix of shape (m, n) to be normalized,
                           where m is the number of data points
                           and n is the number of features.
        gamma (numpy.ndarray): The scales used for batch
        normalization, shape (1, n).
        beta (numpy.ndarray): The offsets used for batch
        normalization, shape (1, n).
        epsilon (float): A small number used to avoid division by zero.

    Returns:
        numpy.ndarray: The normalized Z matrix.
    """
    # calculate the mean of each feature
    mean = np.mean(Z, axis=0)

    # calculate the variance of each feature
    variance = np.var(Z, axis=0)

    # normalize the matrix Z
    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)

    # scale and shift
    Z_scaled_shifted = gamma * Z_normalized + beta

    return Z_scaled_shifted
