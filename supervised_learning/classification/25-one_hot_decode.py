#!/usr/bin/env python3
"""
defines function that converts a one-hot matrix
into a vector of labels
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Parameters:
    -----------
    one_hot : numpy.ndarray
        One-hot encoded matrix with shape (classes, m).
        - classes is the number of classes.
        - m is the number of examples.

    Returns:
    --------
    numpy.ndarray
        A vector of labels with shape (m,), or None on failure.
    """
    # Validate input
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    try:
        # Use np.argmax to find the index of the maximum value
        # along the first axis (rows).
        # This index corresponds to the class label.
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
