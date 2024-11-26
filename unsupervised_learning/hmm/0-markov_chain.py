#!/usr/bin/env python3
"""
Function that determines the probability of a Markov Chain
being in a particular state after a specified number of iterations
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a Markov chain being in a particular state
    after a specified number of iterations.

    Parameters:
        P (numpy.ndarray): Square 2D array representing the transition matrix.
        s (numpy.ndarray): 1D array representing the initial state
        probabilities.
        t (int): Number of iterations.

    Returns:
        numpy.ndarray: Probability of being in a specific state
        after t iterations.
    """
    # Validate inputs
    if not isinstance(P, np.ndarray) or P.ndim != 2 or\
            P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2 or\
            s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int) or t < 0:
        return None

    # Compute the state probabilities after t iterations
    for _ in range(t):
        s = np.matmul(s, P)

    return s
