#!/usr/bin/env python3
"""
Function that calculates a correlation matrix
"""

import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix.

    Parameters:
        C (numpy.ndarray): Covariance matrix of shape (d, d).

    Returns:
        numpy.ndarray: Correlation matrix of shape (d, d).
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Calculate standard deviations
    stddev = np.sqrt(np.diag(C))  # Extract standard deviations from diagonal

    # Prevent division by zero
    stddev_inv = 1 / stddev
    stddev_inv[np.isinf(stddev_inv)] = 0

    # Create correlation matrix
    corr = C * stddev_inv[:, None] * stddev_inv[None, :]

    return corr
