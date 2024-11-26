#!/usr/bin/env python3
"""
Determines the steady state probabilities of a regular Markov Chain.
"""

import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov Chain.

    Parameters:
        P (numpy.ndarray): Square 2D array of shape (n, n)
            - P[i, j] is the probability of transitioning from state
            i to state j
            - n is the number of states in the Markov Chain

    Returns:
        numpy.ndarray of shape (1, n): Steady state probabilities,
        or None on failure
    """
    # Check if P is a valid square matrix
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None

    n, n_check = P.shape
    if n != n_check:
        return None

    # Check if all elements of P are probabilities (between 0 and 1)
    if not np.all((0 <= P) & (P <= 1)):
        return None

    # Check if the rows of P sum to 1
    if not np.allclose(P.sum(axis=1), 1):
        return None

    # Check if P is regular (all elements are strictly positive)
    if not (P > 0).all():
        return None

    # Solve for steady state probabilities
    try:
        # Transpose P for solving (I - P^T)x = 0
        P_transposed = P.T
        A = P_transposed - np.eye(n)
        A[-1] = np.ones(n)
        b = np.zeros(n)
        b[-1] = 1

        # Solve the linear system
        steady_state = np.linalg.lstsq(A, b, rcond=None)[0]

        # Reshape as a row vector
        return steady_state.reshape(1, n)

    except np.linalg.LinAlgError:
        return None
