#!/usr/bin/env python3
"""
Defines a function that calculates the softmax
cross-entropy loss of a prediction
"""

import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.
    Parameters:
    - y: tf.placeholder, the true labels of the input data.
    - y_pred: tf.Tensor, the network's predictions.
    Returns:
    - A tensor containing the loss of the prediction.
    """
    # Calculate the softmax cross-entropy loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
