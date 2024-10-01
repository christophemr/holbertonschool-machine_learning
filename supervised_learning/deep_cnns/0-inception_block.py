#!/usr/bin/env python3
"""
Defines a function that builds an inception block
as described in Going Deeper with Convolutions (2014)
"""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in 'Going Deeper
    with Convolutions' (2014).
    Parameters:
    - A_prev: output from the previous layer
    - filters: a tuple or list containing F1, F3R, F3, F5R, F5,
    FPP respectively
    Returns:
    - The concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Branch 1: 1x1 Convolution
    conv1x1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                              padding='same', activation='relu')(A_prev)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    conv3x3_reduce = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                                     padding='same', activation='relu')(A_prev)
    conv3x3 = K.layers.Conv2D(
      filters=F3, kernel_size=(3, 3),
      padding='same', activation='relu')(conv3x3_reduce)

    # Branch 3: 1x1 Convolution followed by 5x5 Convolution
    conv5x5_reduce = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                                     padding='same', activation='relu')(A_prev)
    conv5x5 = K.layers.Conv2D(
      filters=F5, kernel_size=(5, 5),
      padding='same', activation='relu')(conv5x5_reduce)

    # Branch 4: Max Pooling followed by 1x1 Convolution
    max_pool = K.layers.MaxPooling2D(
      pool_size=(3, 3), strides=(1, 1), padding='same')(A_prev)
    pool_proj = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                                padding='same', activation='relu')(max_pool)

    # Concatenate all the branches
    output = K.layers.concatenate(
      [conv1x1, conv3x3, conv5x5, pool_proj], axis=-1)

    return output
