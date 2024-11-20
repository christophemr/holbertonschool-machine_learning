#!/usr/bin/env python3
"""
Function to calculate the PDF of a Gaussian distribution
"""

import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    Parameters:
        X (numpy.ndarray): Data points, shape (n, d)
        m (numpy.ndarray): Mean of the distribution, shape (d,)
        S (numpy.ndarray): Covariance matrix, shape (d, d)

    Returns:
        numpy.ndarray: PDF values for each data point, shape (n,)
        or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    n, d = X.shape
    if m.shape[0] != d or S.shape != (d, d):
        return None

    try:
        # Determinant and inverse of covariance matrix
        det = np.linalg.det(S)
        if det <= 0:
            return None
        inv_S = np.linalg.inv(S)

        # Normalize constant
        norm_const = 1 / np.sqrt(((2 * np.pi) ** d) * det)

        # Exponent computation
        diff = X - m
        exponent = -0.5 * np.sum(diff @ inv_S * diff, axis=1)

        # PDF values
        P = norm_const * np.exp(exponent)
        # Enforce minimum value of 1e-300
        return np.maximum(P, 1e-300)
    except Exception:
        return None
