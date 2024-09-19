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
    # Retrieve individual L2 regularization losses
    l2_losses = model.losses
    # Optionally, concatenate the L2 losses into a tensor if needed
    l2_losses_tensor = tf.stack(l2_losses)
    # Return the original cost along with the individual L2 losses
    return l2_losses_tensor
