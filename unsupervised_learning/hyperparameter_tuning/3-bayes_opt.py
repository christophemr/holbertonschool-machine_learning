#!/usr/bin/env python3
"""
Creates a class that performs Bayesian optimization on a
noiseless 1D Gaussian process
"""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Represents the Bayesian optimization of a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes the BayesianOptimization class

        Parameters:
            f (callable): The black-box function to be optimized
            X_init (numpy.ndarray): Array of shape (t, 1) of inputs
            already sampled
            Y_init (numpy.ndarray): Array of shape (t, 1) of outputs of the
            black-box function
            bounds (tuple): (min, max) bounds of the search space
            ac_samples (int): Number of samples to analyze during acquisition
            l (float): Length parameter for the kernel
            sigma_f (float): Standard deviation given to the output of the
            black-box function
            xsi (float): Exploration-exploitation factor for acquisition
            minimize (bool): Whether to minimize (True) or maximize (False)
        """
        # Initialize attributes
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        # Generate evenly spaced acquisition sample points
        min_bound, max_bound = bounds
        self.X_s = np.linspace(min_bound, max_bound, ac_samples).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize
