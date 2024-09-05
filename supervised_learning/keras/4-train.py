#!/usr/bin/env python3
"""
Defines a function that trains a model using mini-batch gradient descent
using Keras library
"""


import tensorflow.keras as K  # type: ignore


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Parameters:
    - network: keras.Model, the model to train.
    - data: numpy.ndarray, shape (m, nx), containing the input data.
    - labels: numpy.ndarray, shape (m, classes),
    containing the labels of the data.
    - batch_size: int, size of the batch used for mini-batch gradient descent.
    - epochs: int, number of passes through data
    for mini-batch gradient descent.
    - verbose: bool, determines if output should be printed during training.
    - shuffle: bool, determines whether to shuffle the batches every epoch.

    Returns:
    - History: the History object generated after training the model.
    """
    # Train the model using the fit method
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle)
    # Return the history object
    return history
