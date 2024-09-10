#!/usr/bin/env python3
"""function that normalizes a matrix"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix X using the provided mean and
    standard deviation.

    Args:
        X (numpy.ndarray): The matrix of shape (d, nx) to normalize, where
                           d is the number of data points and nx is the
                           number of features.
        m (numpy.ndarray): The mean of all features of X, of shape (nx,).
        s (numpy.ndarray): The standard deviation of all features of X, of
                           shape (nx,).

    Returns:
        numpy.ndarray: The normalized matrix X.
    """
    # Normalize each feature of X using broadcasting
    normalized = (X - m) / s
    return normalized
