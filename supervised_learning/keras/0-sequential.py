#!/usr/bin/env python3
"""
Defines a function that builds a neural network
using Keras library
"""

import tensorflow.keras as K  # type: ignore


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Parameters:
    - nx: int, number of input features to the network.
    - layers: list, containing the number of nodes
    in each layer of the network.
    - activations: list, containing the activation functions for each layer.
    - lambtha: float, L2 regularization parameter.
    - keep_prob: float, probability that a node will be kept during dropout.

    Returns:
    - model: keras.Model, the Keras model built.
    """
    # Initialize a sequential model using K
    model = K.Sequential()

    # Add the first layer with input shape and regularization
    model.add(K.layers.Dense(units=layers[0],
                             activation=activations[0],
                             kernel_regularizer=K.regularizers.l2(lambtha),
                             input_shape=(nx,)))

    # Add subsequent layers with dropout
    for i in range(1, len(layers)):
        # Add a dropout layer with the specified keep probability
        model.add(K.layers.Dropout(rate=1 - keep_prob))
        # Add a dense layer with regularization and activation
        model.add(K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)))

    return model
