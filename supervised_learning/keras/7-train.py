#!/usr/bin/env python3
"""
Updates function that trains a model using mini-batch gradient descent
to train using learning rate decay with Keras library
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent,
        including analyzing validation data, using early stopping,
        and learning rate decay.

    Parameters:
        network [keras model]: model to train
        data [numpy.ndarray of shape (m, nx)]:
            contains the input data
        labels [one-hot numpy.ndarray of shape (m, classes)]:
            contains labels of data
        batch_size [int]:
            size of batch used for mini-batch gradient descent
        epochs [int]:
            number of passes through data for mini-batch gradient descent
        validation_data:
            data to be analyzed during model training
        early_stopping [boolean]:
            indicates whether early stopping should be used
            early stopping should only be performed if validation_data exists
            early stopping should be based on validation loss
        patience:
            patience used for early stopping
        learning_rate_decay [boolean]:
            indicates whether learning rate decay should be used
            learning rate decay should only be performed
            if validation_data exists
            decay should be performed using inverse time decay
            learning rate should decay in a stepwise fashion after each epoch
            each time the learning rate updates, Keras should print a message
        alpha [float]:
            initial learning rate
        decay_rate [float]:
            decay rate
        verbose [boolean]:
            determines if output should be printed during training
        shuffle [boolean]:
            determines whether to shuffle the batches every epoch

    Returns:
        the History object generated after training the model
    """
    callbacks = []

    # Early Stopping
    if early_stopping and validation_data:
        callbacks.append(
            K.callbacks.EarlyStopping(monitor='val_loss', patience=patience))

    # Learning Rate Decay
    if learning_rate_decay and validation_data:
        def learning_rate(epoch):
            """
            Calculates the learning rate using inverse time decay.

            Formula: initial_learning_rate / (1 + decay_rate * epoch)
            """
            return alpha / (1 + decay_rate * epoch)

        # LearningRateScheduler to apply the decay
        callbacks.append(
            K.callbacks.LearningRateScheduler(learning_rate, verbose=1))

    # Training the model
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks if callbacks else None,
        verbose=verbose,
        shuffle=shuffle
    )

    return history
