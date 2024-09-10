#!/usr/bin/env python3
"""function that creates  mini-batches to be used for training
a neural network using mini-batch gradient descent
"""

import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from the data to be used
    for training with mini-batch gradient descent.

    Args:
        X (numpy.ndarray): Input data of shape (m, nx) where
                           m is the number of data points
                           and nx is the number of features.
        Y (numpy.ndarray): Labels of shape (m, ny) where
                           m is the same number of data points as in X and
                           ny is the number of classes 4 classification tasks.
        batch_size (int): The number of data points in each mini-batch.

    Returns:
        list of tuples: A list of mini-batches,
        each containing a tuple (X_batch, Y_batch).
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    # Initialize a list to store the mini-batches
    mini_batches = []
    # Create mini-batches of size `batch_size`
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
