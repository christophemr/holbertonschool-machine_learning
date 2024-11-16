#!/usr/bin/env python3
"""
Function that performs PCA to reduce dataset dimensions to a specified number.
"""

import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset to reduce its dimensionality to ndim dimensions.

    Parameters:
        X (numpy.ndarray): Dataset of shape (n, d) where
            - n is the number of data points.
            - d is the number of dimensions of each data point.
        ndim (int): The target number of dimensions for the transformed dataset

    Returns:
        numpy.ndarray: Transformed dataset of shape (n, ndim).
    """
    X_centered = X - np.mean(X, axis=0)
    # Perform Singular Value Decomposition (SVD)
    u, s, v = np.linalg.svd(X_centered, full_matrices=False)

    # Select the first `ndim` components
    W = v[:ndim].T  # Weights matrix for top `ndim` components

    # Project the dataset onto the reduced space
    T = np.dot(X_centered, W)

    return T
