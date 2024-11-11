#!/usr/bin/env python3
"""
function that calculates the likelihood of obtaining this data given
various hypothetical probabilities of developing severe side effects
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data given
    various probabilities.

    Parameters:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): Hypothetical probabilities of developing
        severe side effects.

    Returns:
        numpy.ndarray: Likelihood of obtaining the data
        for each probability in P.
    """
    # Input validation
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the binomial coefficient
    factorial = np.math.factorial
    binomial_coeff = factorial(n) / (factorial(x) * factorial(n - x))

    # Calculate the likelihood for each value in P
    likelihoods = binomial_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
