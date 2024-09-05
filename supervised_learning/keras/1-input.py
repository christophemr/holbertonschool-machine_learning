#!/usr/bin/env python3
"""
Defines a function that builds a neural network
using Keras library
"""

import tensorflow.keras as K  # type: ignore


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library using the Functional API.

    Parameters:
    - nx: int, number of input features to the network.
    - layers: list of int, number of nodes in each layer of the network.
    - activations: list of str, activation functions for each layer.
    - lambtha: float, L2 regularization parameter.
    - keep_prob: float, probability that a node will be kept during dropout.

    Returns:
    - model: keras.Model, the Keras model built using the Functional API.
    """
    # Define the input layer with the specified number of features
    inputs = K.Input(shape=(nx,))

    # Initialize the first layer connected to the inputs
    x = K.layers.Dense(units=layers[0],
                       activation=activations[0],
                       kernel_regularizer=K.regularizers.l2(lambtha))(inputs)

    # Add subsequent layers with Dropout
    for i in range(1, len(layers)):
        # Apply Dropout before each subsequent layer except the output layer
        x = K.layers.Dropout(rate=1 - keep_prob)(x)
        x = K.layers.Dense(units=layers[i],
                           activation=activations[i],
                           kernel_regularizer=K.regularizers.l2(lambtha))(x)

    # Define the model with the specified input and output
    model = K.Model(inputs=inputs, outputs=x)

    return model
