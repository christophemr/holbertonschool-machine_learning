#!/usr/bin/env python3
"""
This Module defines a function that creates the forward propagation
graph for a neural network using TensorFlow.
"""
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.
    Args:
        x (tf.placeholder): Placeholder for the input data.
        layer_sizes (list of int): List containing the number of nodes
        in each layer of the network.
        activations (list): List containing the activation functions
        for each layer of the network.
    Returns:
        tf.Tensor: The prediction of the network in tensor form.
    """
    output = x
    for size, activation in zip(layer_sizes, activations):
        output = create_layer(output, size, activation)
    return output
