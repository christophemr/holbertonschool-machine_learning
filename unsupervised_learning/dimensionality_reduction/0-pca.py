#!/usr/bin/env python3
"""
Function that performs principal components analysis (PCA) on dataset
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset to reduce dimensionality while maintaining
    a specified fraction of variance.

    Parameters:
        X (numpy.ndarray): The dataset of shape (n, d), where
            - n is the number of data points.
            - d is the number of dimensions in each data point.
            The dataset is assumed to be zero-centered.
        var (float): The fraction of the total variance that should be
            retained in the reduced-dimensional space. Default is 0.95.

    Returns:
        numpy.ndarray: The weights matrix of shape (d, nd), where nd is the
            number of dimensions required to retain the specified variance.
    """
    # Singular Value Decomposition on the dataset X
    _, s, v = np.linalg.svd(X)
    # Compute the cumulative variance
    variance = np.cumsum(s / np.sum(s))
    # find the number of components
    num_components = np.argmax(variance >= var)
    # Return the weights matrix for the top components
    return v.T[:, :num_components + 1]
