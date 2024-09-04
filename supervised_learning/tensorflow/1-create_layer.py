#!/usr/bin/env python3
"""
Defines a function to create a layer for neural network
"""

import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """
    Creates a new layer in a neural network.

    Parameters:
    - prev: tensor, the output of the previous layer.
    - n: int, the number of nodes in the layer to create.
    - activation: the activation function that the layer should use.

    Returns:
    - The tensor output of the new layer.
    """
    # He et al. initialization using VarianceScaling with fan_avg mode
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    # Create a dense layer with the specified number of nodes,
    # activation function, and initializer
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name='layer')  # type: ignore
    # Pass the output of the previous layer to this new layer
    output = layer(prev)  # type: ignore
    return output
