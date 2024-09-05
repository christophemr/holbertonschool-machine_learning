#!/usr/bin/env python3
"""
Updated function that trains a model using mini-batch gradient descent
to also analyze validation data using Keras library
"""


import tensorflow.keras as K  # type: ignore


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent
    and analyzes validation data.

    Parameters:
    - network: keras.Model, the model to train.
    - data: numpy.ndarray, shape (m, nx), containing the input data.
    - labels: numpy.ndarray, shape (m, classes),
    containing the labels of the data.
    - batch_size: int, size of the batch used for mini-batch gradient descent.
    - epochs: int, number of passes through data
    for mini-batch gradient descent.
    - validation_data: tuple, data to validate the model with, if not None.
    - verbose: bool, determines if output should be printed during training.
    - shuffle: bool, determines whether to shuffle the batches every epoch.

    Returns:
    - History: the History object generated after training the model.
    """
    # Train the model with validation data if provided
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data  # Add validation data
    )
    return history
