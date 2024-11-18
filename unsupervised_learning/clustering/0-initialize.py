#!/usr/bin/env python3
"""
Function that initializes cluster centroids for k-means clustering
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: positive integer for the number of clusters

    Returns:
    - numpy.ndarray of shape (k, d) containing the initialized centroids
      or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    return centroids
