#!/usr/bin/env python3
"""function that sets up the RMSProp optimization
algorithm in TensorFlow
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow.
    Args:
        alpha (float): The learning rate.
        beta2 (float): The RMSProp weight
        (discounting factor for the squared gradients).
        epsilon (float): A small number to avoid division by zero.
    Returns:
        optimizer: A TensorFlow RMSProp optimizer.
    """
    # Create the RMSProp optimizer using TensorFlow
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
    return optimizer
