#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization using Keras.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost (tensor): The original cost of the network without regularization.
    model (keras.Model): Keras model that includes layers
    with L2 regularization.

    Returns:
    tensor: The total cost including L2 regularization.
    """
    l2_losses = cost + model.losses
    return l2_losses
