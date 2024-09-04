#!/usr/bin/env python3
"""
Defines a function that calculates the accuracy of a prediction
for the neural network
"""

import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Parameters:
    - y: tf.placeholder, the true labels of the input data.
    - y_pred: tf.Tensor, the network's predictions.

    Returns:
    - A tensor containing the decimal accuracy of the prediction.
    """
    # Find the predicted class by taking the argmax of y_pred
    # along the last dimension
    correct_predictions = (tf.equal(tf.argmax(y, axis=1),
                                    tf.argmax(y_pred, axis=1)))
    # Calculate the accuracy by taking the mean of correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
