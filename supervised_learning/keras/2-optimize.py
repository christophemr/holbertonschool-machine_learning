#!/usr/bin/env python3
"""
Defines a function that sets up Adam optimizer for Keras model
with categorical crossentropy loss and accuracy metrics
"""

import tensorflow.keras as K  # type: ignore


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a Keras model with categorical crossentropy
    loss and accuracy metrics.

    Parameters:
    - network: keras.Model, the model to optimize.
    - alpha: float, the learning rate.
    - beta1: float, the first Adam optimization parameter.
    - beta2: float, the second Adam optimization parameter.

    Returns:
    - None
    """
    # Create the Adam optimizer with the given parameters
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    # Compile the model with categorical crossentropy loss and accuracy metrics
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Explicitly return None
    return None
