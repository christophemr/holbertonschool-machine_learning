#!/usr/bin/env python3
"""
Function that initializes variables for a Gaussian Mixture Model.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model
    Parameters:
        X (numpy.ndarray): Data set of shape (n, d)
        k (int): Number of clusters
    Returns:
        pi (numpy.ndarray): Priors for each cluster, shape (k,)
        m (numpy.ndarray): Centroid means for each cluster, shape (k, d)
        S (numpy.ndarray): Covariance matrices for each cluster,
        shape (k, d, d)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape
    pi = np.full((k,), 1 / k)
    # Initialize centroids with K-means
    m, _ = kmeans(X, k)
    # Initialize covariance matrices as identity matrices
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
