#!/usr/bin/env python3
"""
Defines a function that builds an identity block
as described in Deep Residual Learning for Image Recognition
(2015)
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in 'Deep Residual Learning
    for Image Recognition' (2015).

    Parameters:
    - A_prev: output from the previous layer
    - filters: a tuple or list containing F11, F3, F12 respectively

    Returns:
    - The activated output of the identity block
    """
    F11, F3, F12 = filters

    # Save the input value
    X_shortcut = A_prev

    # He normal initializer with seed 0 specified as a dictionary
    initializer = {'class_name': 'HeNormal', 'config': {'seed': 0}}

    # First component of main path
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='valid',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=-1)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='valid',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=-1)(X)

    # Add shortcut and pass through activation
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
