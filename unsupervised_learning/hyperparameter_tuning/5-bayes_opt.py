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
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes the BayesianOptimization class
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

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

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function.

        Args:
            iterations (int): Maximum number of iterations to perform.

        Returns:
            X_opt (numpy.ndarray): The optimal point.
            Y_opt (numpy.ndarray): The optimal function value.
        """
        for _ in range(iterations):
            # Get the next best sample location
            X_next, _ = self.acquisition()

            # Check if the next sample is already in the sampled inputs
            if X_next in self.gp.X:
                break

            # Evaluate the black-box function at the new sample
            Y_next = self.f(X_next)

            # Update the Gaussian process with the new sample
            self.gp.update(X_next, Y_next)

        # Identify the optimal point and value
        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        return X_opt, Y_opt
