#!/usr/bin/env python3
"""
Defines a function that builds a modified version of LeNet-5 architecture
using TensorFlow
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lenet5(x, y):
    """
    Builds a modified version of LeNet-5 architecture using TensorFlow

    parameters:
        x [tf.placeholder of shape (m, 28, 28, 1)]:
            contains the input images for the network
            m: number of images
        y [tf.placeholder of shape (m, 10)]:
            contains the one-hot labels for the network

    model layers:
    C1: convolutional layer with 6 kernels of shape (5, 5) with same padding
    P2: max pooling layer with kernels of shape (2, 2) with (2, 2) strides
    C3: convolutional layer with 16 kernels of shape (5, 5) with valid padding
    P4: max pooling layer with kernels of shape (2, 2) with (2, 2) strides
    F5: fully connected layer with 120 nodes
    F6: fully connected layer with 84 nodes
    F7: fully connected softmax output layer with 10 nodes

    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method:
        tf.keras.initializers.VarianceScaling(scale=2.0)
    All hidden layers requiring activation should use the relu
    activation function

    returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
            (default hyperparameters)
        a tensor for the loss of the network
        a tensor for the accuracy of the network
    """
    weights_initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # C1: Convolutional Layer (6 filters, 5x5 kernel, same padding,
    # relu activation)
    C1 = tf.layers.conv2d(
      inputs=x, filters=6, kernel_size=(5, 5), padding='same',
      activation=tf.nn.relu, kernel_initializer=weights_initializer)

    # P2: Max Pooling Layer (2x2 pool size, 2x2 stride)
    P2 = tf.layers.max_pooling2d(inputs=C1, pool_size=(2, 2), strides=(2, 2))

    # C3: Convolutional Layer (16 filters, 5x5 kernel, valid padding,
    # relu activation)
    C3 = tf.layers.conv2d(
      inputs=P2, filters=16, kernel_size=(5, 5),
      padding='valid', activation=tf.nn.relu,
      kernel_initializer=weights_initializer)

    # P4: Max Pooling Layer (2x2 pool size, 2x2 stride)
    P4 = tf.layers.max_pooling2d(inputs=C3, pool_size=(2, 2), strides=(2, 2))

    # Flatten the output for the fully connected layers
    flat = tf.layers.flatten(inputs=P4)

    # F5: Fully Connected Layer with 120 nodes, relu activation
    F5 = tf.layers.dense(inputs=flat, units=120, activation=tf.nn.relu,
                         kernel_initializer=weights_initializer)

    # F6: Fully Connected Layer with 84 nodes, relu activation
    F6 = tf.layers.dense(inputs=F5, units=84, activation=tf.nn.relu,
                         kernel_initializer=weights_initializer)

    # F7: Fully Connected Output Layer with 10 nodes (for classification)
    F7 = tf.layers.dense(
      inputs=F6, units=10, kernel_initializer=weights_initializer)

    # Softmax output
    softmax_output = tf.nn.softmax(F7)

    # Loss function (Cross Entropy)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(logits=F7, labels=y))

    # Adam optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy calculation
    correct_predictions = tf.equal(tf.argmax(F7, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return softmax_output, optimizer, loss, accuracy
