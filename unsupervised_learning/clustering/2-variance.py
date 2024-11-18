#!/usr/bin/env python3
"""
Function that calculates the total intra-cluster variance for a dataset
"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.

    Parameters:
        X (numpy.ndarray): The dataset of shape (n, d)
        C (numpy.ndarray): The centroids of shape (k, d)

    Returns:
        var (float): Total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Calculate distances from each point
    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2) ** 2

    # Sum up the minimum distances
    var = np.sum(np.min(distances, axis=1))
    return var
