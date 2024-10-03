#!/usr/bin/env python3
"""
Defines a function that builds a transition layer
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in 'Densely
    Connected Convolutional Networks' (2017).

    Parameters:
    - X: output from the previous layer
    - nb_filters: integer, number of filters in the input tensor X
    - compression: compression factor for the transition layer

    Returns:
    - The output of the transition layer
    - The updated number of filters after compression
    """

    # He normal initializer with seed 0
    initializer = K.initializers.he_normal(seed=0)

    # Calculate the number of filters after compression
    nb_filters = int(nb_filters * compression)

    # Batch Normalization + ReLU
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation('relu')(X)

    # 1x1 Convolution to reduce the number of filters
    X = K.layers.Conv2D(
      filters=nb_filters, kernel_size=(1, 1), padding='same',
      kernel_initializer=initializer)(X)

    # Average Pooling to downsample the feature maps
    X = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=(2, 2), padding='valid')(X)

    return X, nb_filters
