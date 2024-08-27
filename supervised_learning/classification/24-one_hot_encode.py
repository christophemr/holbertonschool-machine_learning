#!/usr/bin/env python3

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Parameters:
    -----------
    Y : numpy.ndarray
        Array of shape (m,) containing numeric class labels.
        m is the number of examples.
    classes : int
        The maximum number of classes found in Y.

    Returns:
    --------
    numpy.ndarray
        One-hot encoded matrix with shape (classes, m),
        or None on failure.
    """
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= 0:
        return None
    if np.max(Y) >= classes:
        return None
    try:
        # Initialize a zero matrix of shape (classes, m)
        one_hot = np.zeros((classes, Y.shape[0]))
        # place 1s in the correct positions
        one_hot[Y, np.arange(Y.shape[0])] = 1

        return one_hot
    except Exception:
        return None
