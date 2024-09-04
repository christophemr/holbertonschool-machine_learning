#!/usr/bin/env python3
"""
Defines a function that creates the training operation
for the neural network
"""

import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network using gradient descent.

    Parameters:
    - loss: tf.Tensor, the loss of the network's prediction.
    - alpha: float, the learning rate.

    Returns:
    - An operation that trains the network using gradient descent.
    """
    # Create a gradient descent optimizer with the specified learning rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    # Create the training operation that minimizes the loss
    train_op = optimizer.minimize(loss)

    return train_op
