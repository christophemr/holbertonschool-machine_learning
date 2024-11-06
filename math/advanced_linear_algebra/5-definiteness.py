#!/usr/bin/env python3
"""
Function that calculates the definiteness of a matrix
"""

import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a matrix

    Parameters:
        matrix (numpy.ndarray): matrix whose definiteness should be calculated

    Returns:
        str: The definiteness of the matrix ("Positive definite",
             "Positive semi-definite", "Negative semi-definite",
             "Negative definite","Indefinite") or None if the matrix is invalid
    """
    # Input validation
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None  # Not a valid square matrix

    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Check the definiteness based on eigenvalues
    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"

    return None
