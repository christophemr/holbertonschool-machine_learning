#!/usr/bin/env python3
"""
Function that calculates the posterior probability for the
various hypothetical probabilities of developing severe side effects
given the data
"""

import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability of each probability
    in P given x and n.

    Parameters:
        x (int): Number of patients that develop severe side effects.
        n (int): Total number of patients observed.
        P (numpy.ndarray): Probabilities of developing severe side effects.
        Pr (numpy.ndarray): Prior beliefs of P.

    Returns:
        numpy.ndarray: Posterior probability of each probability in P.
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
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood as the binomial distribution
    factorial = np.math.factorial
    fact_coefficient = factorial(n) / (factorial(n - x) * factorial(x))
    likelihood = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))
    # Calculate the intersection
    intersection = likelihood * Pr
    # Calculate the marginal probability
    marginal = np.sum(intersection)
    # Calculate the posterior probability
    posterior = intersection / marginal
    return posterior
