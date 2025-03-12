#!/usr/bin/env python3
"""
Compute the policy with a weight of a matrix.
"""

import numpy as np


def policy(matrix, weight):
    """
    Compute the policy with a weight of a matrix.

    Args:
        matrix (numpy.ndarray): A 1D numpy array representing the state.
        weight (numpy.ndarray): A 2D numpy array representing the weights.

    Returns:
        numpy.ndarray: A 1D numpy array representing the policy probabilities.
    """
    # Calculate the dot product of the state and weight matrix
    dot_product = np.dot(matrix, weight)

    # Apply the softmax function to obtain probabilities
    exp_values = np.exp(dot_product - np.max(dot_product))
    policy = exp_values / np.sum(exp_values)

    return policy
