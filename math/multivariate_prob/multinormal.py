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

        self.mean = np.mean(data, axis=1, keepdims=True)  # Shape (d, 1)
        data_centered = data - self.mean  # Centered data
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)
