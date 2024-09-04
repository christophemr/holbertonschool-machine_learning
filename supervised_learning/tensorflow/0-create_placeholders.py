#!/usr/bin/env python3
"""
Defines a function to return two placeholders for the neural network
"""

import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """
    Creates two placeholders, x and y, for the neural network.

    Parameters:
    - nx: int, number of feature columns in our data.
    - classes: int, number of classes in our classifier.

    Returns:
    - x: placeholder for input data, of shape [None, nx].
    - y: placeholder for the labels, of shape [None, classes].
    """
    # Placeholder for input data (features)
    x = tf.placeholder(dtype=tf.float32, shape=[None, nx], name='x')
    # Placeholder for input labels (one-hot)
    y = tf.placeholder(dtype=tf.float32, shape=[None, classes], name='y')
    return x, y
