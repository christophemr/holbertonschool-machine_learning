#!/usr/bin/env python3
"""
Defines a function that creates a TensorFlow layer
that includes L2 Regularization
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer that includes L2 regularization

    Parameters:
    prev (tensor): Output of the previous layer.
    n (int): Number of nodes in the new layer.
    activation (function): Activation function for the new layer.
    lambtha (float): L2 regularization parameter.

    Returns:
    tensor: Output of the new layer.
    """
    l2_reg = tf.keras.regularizers.L2(lambtha)

    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
      mode='fan_avg')

    # Create the dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=l2_reg
    )

    # Return the output of the layer
    return layer(prev)
