#!/usr/bin/env python3
"""
function that updates a variable
using RMSProp optimization algorithm
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.
    Args:
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight.
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of the variable.
        s (numpy.ndarray): The previous second moment of the variable.
    Returns:
        tuple: The updated variable and the new second moment.
    """
    # Update the second moment estimate
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    # Update the variable
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
