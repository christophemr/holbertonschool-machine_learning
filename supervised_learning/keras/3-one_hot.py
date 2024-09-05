#!/usr/bin/env python3
"""
Defines a function that converts a label vector into a one-hot matrix
using Keras library
"""

import tensorflow.keras as K  # type: ignore


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Parameters:
    - labels: array-like, the label vector to convert.
    - classes: int, the number of classes. If None,
    it will infer from the data.

    Returns:
    - The one-hot encoded matrix as a numpy array.
    """
    # Convert the labels to a one-hot matrix
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)

    return one_hot_matrix
