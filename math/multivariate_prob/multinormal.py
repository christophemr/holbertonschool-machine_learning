#!/usr/bin/env python3
"""
 class MultiNormal that represents a Multivariate Normal distribution
"""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes a MultiNormal instance.

        Parameters:
            data (numpy.ndarray): A dataset of shape (d, n), where
                                  d is the number of dimensions,
                                  and n is the number of data points.

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If n is less than 2.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        _, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a given data point.

        Parameters:
            x (numpy.ndarray): Shape (d, 1) containing the data point.

        Returns:
            float: PDF value.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != self.mean.shape:
            raise ValueError(f"x must have the shape {self.mean.shape}")

        d = self.mean.shape[0]

        # Calculate the determinant and inverse of the covariance matrix
        cov_det = np.linalg.det(self.cov)
        if cov_det == 0:
            raise ValueError(
                "Covariance matrix is singular, cannot calculate PDF")
        cov_inv = np.linalg.inv(self.cov)

        # Calculate the normalization factor
        norm_factor = 1 / (np.sqrt((2 * np.pi) ** d * cov_det))

        # Calculate the exponent
        x_centered = x - self.mean
        exponent = -0.5 * np.dot(np.dot(x_centered.T, cov_inv), x_centered)

        # Return the PDF value
        return float(norm_factor * np.exp(exponent))
