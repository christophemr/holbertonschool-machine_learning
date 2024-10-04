#!/usr/bin/env python3
"""
Defines a function that builds a DenseNet-121 network
"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in 'Densely Connected
    Convolutional Networks' (2017).

    Parameters:
    - growth_rate: integer, the growth rate for the dense blocks.
    - compression: float, the compression factor for the transition layers.

    Returns:
    - model: the Keras model for DenseNet-121.
    """

    # Initialize weights using He normal initialization
    init = K.initializers.he_normal(seed=0)
    activation = K.activations.relu

    # Input Layer
    X_input = K.Input(shape=(224, 224, 3))

    # Initial Batch Norm + ReLU + Conv Layer
    X = K.layers.BatchNormalization(axis=3)(X_input)
    X = K.layers.Activation(activation)(X)
    X = K.layers.Conv2D(filters=64,
                        kernel_size=(7, 7),
                        padding='same',
                        strides=(2, 2),
                        kernel_initializer=init)(X)

    # Max Pooling Layer
    X = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(X)

    # Dense Block 1: 6 layers followed by a transition layer
    X, nb_filters = dense_block(X, 64, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2: 12 layers followed by a transition layer
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3: 24 layers followed by a transition layer
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4: 16 layers, no transition layer afterward
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Global Average Pooling
    X = K.layers.AveragePooling2D(
      pool_size=(7, 7), strides=(1, 1), padding='valid')(X)

    # Output Softmax Layer for 1000 classes
    X = K.layers.Dense(1000,
                       activation='softmax',
                       kernel_initializer=init)(X)

    # Create the Keras model
    model = K.Model(inputs=X_input, outputs=X)

    return model
