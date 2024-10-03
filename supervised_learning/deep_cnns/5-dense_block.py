#!/usr/bin/env python3
"""
Defines a function that builds a dense block
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in 'Densely Connected
    Convolutional Networks' (2017).
    Parameters:
    - X: output from the previous layer
    - nb_filters: integer, number of filters in the input tensor X
    - growth_rate: growth rate for the dense block
    - layers: number of layers within the dense block

    Returns:
    - The concatenated output of all layers in the dense block
    - The updated number of filters after the block
    """

    # He normal initializer with seed 0
    initializer = K.initializers.he_normal(seed=0)

    # Iterate over the number of layers in the dense block
    for i in range(layers):
        # Batch Normalization + ReLU + 1x1 Convolution
        X1 = K.layers.BatchNormalization(axis=-1)(X)
        X1 = K.layers.Activation('relu')(X1)
        X1 = K.layers.Conv2D(
          filters=4 * growth_rate, kernel_size=(1, 1), padding='same',
          kernel_initializer=initializer)(X1)

        # Batch Normalization + ReLU + 3x3 Convolution
        X1 = K.layers.BatchNormalization(axis=-1)(X1)
        X1 = K.layers.Activation('relu')(X1)
        X1 = K.layers.Conv2D(
          filters=growth_rate, kernel_size=(3, 3), padding='same',
          kernel_initializer=initializer)(X1)

        # Concatenate the input (X) with the output (X1)
        X = K.layers.Concatenate(axis=-1)([X, X1])

        # Update the number of filters after concatenation
        nb_filters += growth_rate

    return X, nb_filters
