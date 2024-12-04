#!/usr/bin/env python3
"""
Creates class that represents a noiseless 1D Gaussian process
"""

import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes the GaussianProcess class.

        Parameters:
            X_init (numpy.ndarray): Inputs already sampled with the
            black-box function
            Y_init (numpy.ndarray): Outputs of the black-box function for
            each input in X_init
            l (float): Length parameter for the kernel
            sigma_f (float): Standard deviation for the output of the
            black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the RBF kernel.

        Parameters:
            X1 (numpy.ndarray): Matrix of shape (m, 1)
            X2 (numpy.ndarray): Matrix of shape (n, 1)

        Returns:
            numpy.ndarray: Covariance kernel matrix of shape (m, n)
        """
        sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
            np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a Gaussian
        process

        Parameters:
            X_s (numpy.ndarray): Points to predict, shape (s, 1)

        Returns:
            tuple: (mu, sigma)
                mu (numpy.ndarray): Mean of each point in X_s, shape (s,)
                sigma (numpy.ndarray): Variance of each point in X_s, shape(s,)
        """
        # Covariance between X and X_s
        K_s = self.kernel(self.X, X_s)

        # Covariance between X_s and X_s
        K_ss = self.kernel(X_s, X_s)

        # Inverse of the kernel matrix for X
        K_inv = np.linalg.inv(self.K)

        # Predictive mean
        mu_s = K_s.T.dot(K_inv).dot(self.Y).flatten()

        # Predictive covariance
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma_s = np.diag(cov_s)
        return mu_s, sigma_s
