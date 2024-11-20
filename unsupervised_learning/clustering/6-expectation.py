#!/usr/bin/env python3
"""
Function to perform the expectation step in the EM algorithm 4 a GMM
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm 4 a GMM.

    Parameters:
        X (numpy.ndarray): Data set of shape (n, d)
        pi (numpy.ndarray): Priors for each cluster, shape (k,)
        m (numpy.ndarray): Centroid means 4 each cluster, shape (k, d)
        S (numpy.ndarray): Covariance matrices 4 each cluster,
        shape (k, d, d)

    Returns:
        g (numpy.ndarray): Posterior probabilities 4 each data point
        in each cluster, shape (k, n)
        l (float): Total log likelihood
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None
    if pi.shape[0] != m.shape[0] or m.shape[0] != S.shape[0] or\
            m.shape[1] != X.shape[1]:
        return None, None

    try:
        k = pi.shape[0]
        n, d = X.shape

        # Calculate the likelihood 4 each cluster
        likelihoods = np.array([pi[i] * pdf(X, m[i], S[i]) for i in range(k)])

        # Posterior probabilities (responsibilities)
        g = likelihoods / likelihoods.sum(axis=0, keepdims=True)

        # Total log likelihood
        log = np.sum(np.log(likelihoods.sum(axis=0)))

        return g, log
    except Exception:
        return None, None
