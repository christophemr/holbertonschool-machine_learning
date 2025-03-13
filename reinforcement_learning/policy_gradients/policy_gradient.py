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


def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo policy gradient based on a state
    and a weight matrix.

    Args:
        state (numpy.ndarray): A 1D numpy array representing the
        current observation of the environment.
        weight (numpy.ndarray): A 2D numpy array representing the weights.

    Returns:
        tuple: The action (int) and the gradient (numpy.ndarray).
    """
    # Compute the policy probabilities
    probabilities = policy(state, weight)

    # Choose an action based on the probabilities
    action = np.random.choice(len(probabilities), p=probabilities)

    # Initialize the gradient matrix
    gradient = np.zeros_like(weight)

    # Calculate the gradient
    for i in range(len(probabilities)):
        indicator = 1 if i == action else 0
        gradient[:, i] = state * (indicator - probabilities[i])

    return action, gradient
