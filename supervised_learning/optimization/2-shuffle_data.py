#!/usr/bin/env python3
"""function that shuffles the data point
in two matrices the same way"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices X and Y in the same way.

    Args:
        X (numpy.ndarray): The first matrix of shape (m, nx) to shuffle,
                           where m is the number of data points and nx
                           is the number of features in X.
        Y (numpy.ndarray): The second matrix of shape (m, ny) to shuffle,
                           where m is the same number of data points as in X
                           and ny is the number of features in Y.
    Returns:
        tuple: The shuffled X and Y matrices.
    """
    m = X.shape[0]
    # Generate a random permutation of indices from 0 to m-1
    permutation = np.random.permutation(m)
    # Apply the permutation to shuffle X and Y
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled
