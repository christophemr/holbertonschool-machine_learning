#!/usr/bin/env python3
"""
function that updates a variable in place
using the Adam optimization algorithm
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The weight for the first moment.
        beta2 (float): The weight for the second moment.
        epsilon (float): A small number to avoid division by zero.
        var (numpy.ndarray): The variable to be updated.
        grad (numpy.ndarray): The gradient of the variable.
        v (numpy.ndarray): The previous first moment of the variable.
        s (numpy.ndarray): The previous second moment of the variable.
        t (int): The time step used for bias correction.

    Returns:
        tuple: The updated variable, the new first moment,
        and the new second moment.
    """
    # Update the first moment estimate
    v = beta1 * v + (1 - beta1) * grad

    # Update the second moment estimate
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Correct the bias in the first moment
    v_cor = v / (1 - beta1 ** t)

    # Correct the bias in the second moment
    s_cor = s / (1 - beta2 ** t)

    # Update the variable
    var = var - alpha * v_cor / (np.sqrt(s_cor) + epsilon)

    return var, v, s
