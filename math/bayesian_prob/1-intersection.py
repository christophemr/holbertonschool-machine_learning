#!/usr/bin/env python3
"""
Function that calculates the intersection of obtaining this data
with the various hypothetical probabilities of developing severe side effects
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


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining data with the
    various probabilities.

    Parameters:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): Probabilities of developing severe side effects.
        Pr (numpy.ndarray): Prior beliefs of P.

    Returns:
        numpy.ndarray: Intersection of obtaining x and n with
        each probability in P.
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood
    likelihoods = likelihood(x, n, P)
    # Calculate the intersection
    intersections = likelihoods * Pr
    return intersections
