#!/usr/bin/env python3
"""
Creates a class that performs Bayesian optimization on a
noiseless 1D Gaussian process
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Represents the Bayesian optimization of a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True, redundancy_threshold=1e-3):
        """
        Initializes the BayesianOptimization class
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, 50).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.redundancy_threshold = redundancy_threshold

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement

        Returns:
            next_sample (numpy.ndarray): The next best sample point
            expected_improvement (numpy.ndarray): Expected Improvement for
            each sample point
        """
        # Predict the mean and standard deviation for the sample points
        mean, std_dev = self.gp.predict(self.X_s)

        # Determine the best current sample value
        if self.minimize:
            best_sample_value = np.min(self.gp.Y)
            improve = best_sample_value - mean - self.xsi
        else:
            best_sample_value = np.max(self.gp.Y)
            improve = mean - best_sample_value - self.xsi

        # Handle division by zero in standard deviation
        with np.errstate(divide='ignore'):
            score = improve / std_dev
            expected = (improve * norm.cdf(score) + std_dev * norm.pdf(score))

            expected[std_dev == 0.0] = 0.0

        # Find the sample point with the maximum Expected Improvement
        next_sample = self.X_s[np.argmax(expected)]

        return next_sample, expected

    def is_redundant(self, X_next, X_samples, threshold=1e-3):
        """Checks if a sample point is redundant based on threshold."""
        return np.any(np.linalg.norm(X_next - X_samples, axis=1) < threshold)

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function.

        Args:
            iterations (int): Maximum number of iterations to perform.

        Returns:
            X_opt (numpy.ndarray): The optimal point.
            Y_opt (numpy.ndarray): The optimal function value.
        """

        for i in range(iterations):
            # Calculate next sample point
            X_next, EI = self.acquisition()

            # Stop if next sample point is redundant
            if self.is_redundant(X_next, self.gp.X):
                continue
            # Reset redundant counter and update GP
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        # Find optimal point
        if self.minimize:
            best_idx = np.argmin(self.gp.Y)
        else:
            best_idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[best_idx]
        Y_opt = self.gp.Y[best_idx]

        return X_opt, Y_opt
