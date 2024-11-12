#!/usr/bin/env python3
"""
Function that calculates the mean and covariance of a data set
"""

import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Parameters:
        X (numpy.ndarray): Shape (n, d) containing the data set.

    Returns:
        tuple: mean (1, d) and covariance matrix (d, d).
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate the mean
    mean = np.mean(X, axis=0, keepdims=True)  # Shape (1, d)
    # Center the data (X - mean)
    X_centered = X - mean
    # Calculate the covariance matrix
    cov = np.dot(X_centered.T, X_centered) / (n - 1)  # Shape (d, d)
    return mean, cov
