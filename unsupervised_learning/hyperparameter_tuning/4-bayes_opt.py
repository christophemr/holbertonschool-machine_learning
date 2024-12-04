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
            improvement = best_sample_value - mean - self.xsi
        else:
            best_sample_value = np.max(self.gp.Y)
            improvement = mean - best_sample_value - self.xsi

        # Handle division by zero in standard deviation
        with np.errstate(divide='ignore'):
            standard_score = improvement / std_dev
            expected_improvement = (
                improvement * norm.cdf(standard_score)
                + std_dev * norm.pdf(standard_score)
            )

            expected_improvement[std_dev == 0.0] = 0.0

        # Find the sample point with the maximum Expected Improvement
        next_sample = self.X_s[np.argmax(expected_improvement)]

        return next_sample, expected_improvement
