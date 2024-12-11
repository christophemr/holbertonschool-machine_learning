#!/usr/bin/env python3
"""Creates a sparse autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder.

    Parameters:
    - input_dims: int, dimensions of the model input.
    - hidden_layers: list of ints, number of nodes for each hidden
    layer in the encoder.
    - latent_dims: int, dimensions of the latent space representation.
    - lambtha: float, L1 regularization parameter on the encoded output.

    Returns:
    - encoder: encoder model.
    - decoder: decoder model.
    - auto: sparse autoencoder model.
    """
    # Input layer
    inputs = keras.Input(shape=(input_dims,))

    # Build the encoder
    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)

    # Latent space with L1 regularization
    encoded = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.L1(lambtha)
    )(encoded)

    # Build the decoder (reverse of encoder)
    decoded = encoded
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)

    # Output layer
    decoded = (
        keras.layers.Dense(input_dims, activation='sigmoid')(decoded))

    # Models
    encoder = keras.Model(inputs, encoded, name="encoder")
    decoder_input = keras.Input(shape=(latent_dims,))
    decoder_output = decoder_input
    for nodes in reversed(hidden_layers):
        decoder_output = (
            keras.layers.Dense(nodes, activation='relu')(decoder_output))
    decoder_output = (
        keras.layers.Dense(input_dims, activation='sigmoid')(decoder_output))
    decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    # Autoencoder
    auto = keras.Model(
        inputs, decoder(encoder(inputs)), name="autoencoder")
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
