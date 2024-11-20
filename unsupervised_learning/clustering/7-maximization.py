#!/usr/bin/env python3
"""
Function to perform the maximization step in the EM algorithm for a GMM
"""

import numpy as np


def maximization(X, g):
    """
    Maximization step for the EM algorithm for a GMM.

    Parameters:
        X (numpy.ndarray): Data set of shape (n, d)
        g (numpy.ndarray): Posterior probabilities, shape (k, n)

    Returns:
        pi (numpy.ndarray): Updated priors for each cluster, shape (k,)
        m (numpy.ndarray): Updated centroid means for each cluster, shape (k, d)
        S (numpy.ndarray): Updated covariance matrices for each cluster, shape (k, d, d)
    """
    try:
        # Validate inputs
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None
        if not isinstance(g, np.ndarray) or g.ndim != 2:
            return None, None, None

        n, d = X.shape
        k, n_g = g.shape
        if n != n_g:
            return None, None, None

        # Compute the priors
        pi = np.sum(g, axis=1) / n

        # Compute the means
        m = np.dot(g, X) / np.sum(g, axis=1)[:, None]

        # Compute the covariance matrices
        S = np.zeros((k, d, d))
        for i in range(k):
            diff = X - m[i]
            weighted_diff = g[i, :, None, None] * np.einsum('ij,ik->ijk', diff, diff)
            S[i] = weighted_diff.sum(axis=0) / g[i].sum()

        return pi, m, S
    except Exception as e:
        print(f"Error during computation: {e}")
        return None, None, None
