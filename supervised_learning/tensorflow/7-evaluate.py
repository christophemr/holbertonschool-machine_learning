#!/usr/bin/env python3
"""
Defines a function that evaluates output of
neural network classifier
"""

import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Parameters:
    - X: np.ndarray, input data to evaluate.
    - Y: np.ndarray, one-hot labels for X.
    - save_path: str, location to load the model from.

    Returns:
    - A tuple containing the network's prediction,
    accuracy, and loss, respectively.
    """
    # Start a new session
    with tf.Session() as sess:
        # Load the saved model
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        # Retrieve the necessary tensors from the graph's collections
        # Input placeholder
        x = tf.get_collection('x')[0]
        # Labels placeholder
        y = tf.get_collection('y')[0]
        # Prediction tensor
        y_pred = tf.get_collection('y_pred')[0]
        # Accuracy tensor
        accuracy = tf.get_collection('accuracy')[0]
        # Loss tensor
        loss = tf.get_collection('loss')[0]

        # Evaluate the network's predictions, accuracy, and loss
        predictions = sess.run(y_pred, feed_dict={x: X, y: Y})
        acc = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss_value = sess.run(loss, feed_dict={x: X, y: Y})

    return predictions, acc, loss_value
