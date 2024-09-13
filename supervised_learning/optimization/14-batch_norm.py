#!/usr/bin/env python3
"""function tht creates a batch normalization layer in TensorFlow."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
    prev (tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation (function): The activation function to be used on the output
        of the layer.

    Returns:
    tensor: The activated output for the layer.
    """
    # Initialize weights with VarianceScaling initializer
    weight_initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Create the Dense layer without activation (to match your logic)
    dense_layer = tf.keras.layers.Dense(
      units=n, kernel_initializer=weight_initializer)
    layer_output = dense_layer(prev)

    # Manual batch normalization with gamma and beta initialization
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    # Compute mean and variance across the 0 axis (batch axis)
    mean, variance = tf.nn.moments(layer_output, axes=0)

    # Apply batch normalization
    normalized_output = tf.nn.batch_normalization(
      layer_output, mean, variance, beta, gamma, 1e-7)

    # Apply the activation function
    return activation(normalized_output)
