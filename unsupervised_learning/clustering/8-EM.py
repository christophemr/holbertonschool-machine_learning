#!/usr/bin/env python3
"""
Function that performs Expectation-Maximization for
Gaussian Mixture Model
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the Expectation-Maximization algorithm for a GMM.

    Parameters:
        X (numpy.ndarray): Dataset of shape (n, d)
        k (int): Number of clusters
        iterations (int): Maximum number of iterations
        tol (float): Tolerance for log likelihood difference
        verbose (bool): If True, prints log likelihood every 10 iterations

    Returns:
        pi (numpy.ndarray): Priors for each cluster, shape (k,)
        m (numpy.ndarray): Centroid means, shape (k, d)
        S (numpy.ndarray): Covariance matrices, shape (k, d, d)
        g (numpy.ndarray): Probabilities for each data point in each cluster,
        shape (k, n)
        l (float): Log likelihood of the model
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None, None, None, None
        if not isinstance(k, int) or k <= 0:
            return None, None, None, None, None
        if not isinstance(iterations, int) or iterations <= 0:
            return None, None, None, None, None
        if not isinstance(tol, (int, float)) or tol < 0:
            return None, None, None, None, None

        # Initialize the parameters
        pi, m, S = initialize(X, k)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        l_prev = 0  # Initialize previous log likelihood
        for i in range(iterations):
            # Expectation Step
            g, l = expectation(X, pi, m, S)
            if g is None or l is None:
                return None, None, None, None, None

            # Verbose logging
            if verbose and (i % 10 == 0 or i == iterations - 1 or
                            abs(l - l_prev) <= tol):
                print(f"Log Likelihood after {i} iterations: {l:.5f}")

            # Convergence check
            if abs(l - l_prev) <= tol:
                break
            l_prev = l

            # Maximization Step
            pi, m, S = maximization(X, g)
            if pi is None or m is None or S is None:
                return None, None, None, None, None

        if verbose:
            print(f"Log Likelihood after {i} iterations: {l:.5f}")

        return pi, m, S, g, l

    except Exception as e:
        print(f"Error during computation: {e}")
        return None, None, None, None, None
