#!/usr/bin/env python3
"""
function that calculates the normalization (standardization)
constants of a matrix:
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the mean and standard deviation of each feature in the matrix X.

    Args:
        X (numpy.ndarray): A matrix of shape (m, nx) to normalize,
                           where m is the number of data points and
                           nx is the number of features.

    Returns:
        tuple: (mean, std) where mean is the mean of each feature,
               and std is the standard deviation of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (mean, std)
