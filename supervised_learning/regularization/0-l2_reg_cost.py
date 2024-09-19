#!/usr/bin/env python3
"""
Module to calculate L2 regularization cost.
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    Parameters:
    cost (float): The original cost of the network without regularization.
    lambtha (float): The regularization parameter.
    weights (dict): A dictionary containing the weights of
    the network's layers.
    L (int): The number of layers in the neural network.
    m (int): The number of data points.
    Returns:
    float: The cost of the network accounting for L2 regularization.
    """
    l2_cost = cost
    # Sum the L2 norm (squared weights) across all layers
    for i in range(1, L + 1):
        l2_cost += lambtha / (2 * m) * np.sum(np.square(weights['W' + str(i)]))
    return l2_cost
