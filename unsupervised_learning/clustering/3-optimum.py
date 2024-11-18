#!usr/bin/env python 3
"""
Function to test the optimum number of clusters by variance
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance.

    Parameters:
        X (numpy.ndarray): The dataset of shape (n, d)
        kmin (int): Minimum number of clusters to check for (inclusive)
        kmax (int): Maximum number of clusters to check for (inclusive)
        iterations (int): Maximum number of iterations for K-means

    Returns:
        results (list): Outputs of K-means for each cluster size
        d_vars (list): Difference in variance from the smallest cluster size
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0 or not isinstance(
            iterations, int) or iterations <= 0:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= kmin):
        return None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    if kmax - kmin < 1:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        results.append((C, clss))
        variances.append(variance(X, C))

    # Calculate differences in variances from the smallest cluster size
    d_vars = [variances[0] - v for v in variances]

    return results, d_vars
