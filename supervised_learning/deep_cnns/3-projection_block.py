#!/usr/bin/env python3
"""
Defines a function that builds a projection block
using Keras
"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in 'Deep Residual Learning for
    Image Recognition' (2015).

    Parameters:
    - A_prev: output from the previous layer
    - filters: a tuple or list containing F11, F3, F12 respectively
    - s: stride of the first convolution in both the main path and the shortcut

    Returns:
    - The activated output of the projection block
    """
    F11, F3, F12 = filters

    # First component of main path
    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=s,
                        padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0)
                        )(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of main path
    X = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0)
                        )(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of main path
    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0)
                        )(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Shortcut path
    X_shortcut = K.layers.Conv2D(
      filters=F12, kernel_size=(1, 1), strides=s, padding='same',
      kernel_initializer=K.initializers.he_normal(seed=0))(A_prev)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Add shortcut and pass through activation
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
