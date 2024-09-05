#!/usr/bin/env python3
"""
Updated function that trains a model using mini-batch gradient descent
to train using early stopping with Keras library
"""

import tensorflow.keras as K  # type: ignore


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent, analyzes validation data,
    and optionally applies early stopping.

    Parameters:
    - network: keras.Model, the model to train.
    - data: numpy.ndarray, shape (m, nx), containing the input data.
    - labels: numpy.ndarray, shape (m, classes),
    containing the labels of the data.
    - batch_size: int, size of the batch used for mini-batch gradient descent.
    - epochs: int, number of passes through data
    for mini-batch gradient descent.
    - validation_data: tuple, data to validate the model with, if not None.
    - early_stopping: bool, indicates whether early stopping should be used.
    - patience: int, patience used for early stopping.
    - verbose: bool, determines if output should be printed during training.
    - shuffle: bool, determines whether to shuffle the batches every epoch.

    Returns:
    - History: the History object generated after training the model.
    """
    # Prepare the list of callbacks
    callbacks = []

    # Add EarlyStopping callback if early_stopping is enabled
    # and validation_data is provided
    if early_stopping and validation_data is not None:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            # Number of epochs with no improvement after which
            # training will be stopped
            patience=patience,
        )
        callbacks.append(early_stopping_callback)

    # Train the model
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,  # Add validation data if provided
        callbacks=callbacks               # Include the callbacks list
    )

    return history
