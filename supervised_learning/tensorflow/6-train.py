#!/usr/bin/env python3
"""
Defines a function that builds, trains, and saves
neural network classifier
"""

import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Builds, trains, and saves a neural network classifier.

    Parameters:
    - X_train: np.ndarray, training input data.
    - Y_train: np.ndarray, training labels.
    - X_valid: np.ndarray, validation input data.
    - Y_valid: np.ndarray, validation labels.
    - layer_sizes: list of int, number of nodes in each layer of the network.
    - activations: list of activation functions for each layer of the network.
    - alpha: float, learning rate.
    - iterations: int, number of iterations to train over.
    - save_path: str, path where the model should be saved.

    Returns:
    - The path where the model was saved.
    """
    # Create placeholders for input data and labels
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Build the forward propagation graph
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Create the training operation
    train_op = create_train_op(loss, alpha)

    # Add elements to collections for easy access
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialize global variables
    init = tf.global_variables_initializer()

    # Create a saver to save the model
    saver = tf.train.Saver()

    # Start the TensorFlow session
    with tf.Session() as sess:
        sess.run(init)

        # Training loop
        for i in range(iterations + 1):
            # Run training operation
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

            # Print metrics every 100 iterations, at 0,
            # and at the last iteration
            if i % 100 == 0 or i == 0 or i == iterations:
                # Calculate training cost and accuracy
                train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
                train_accuracy = (sess.run(accuracy, feed_dict={x: X_train,
                                                                y: Y_train}))

                # Calculate validation cost and accuracy
                valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
                valid_accuracy = (sess.run(accuracy, feed_dict={x: X_valid,
                                                                y: Y_valid}))

                # Print the metrics
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

        # Save the model after training
        saved_path = saver.save(sess, save_path)

    return saved_path
