#!/usr/bin/env python3
"""Defines a function to create a variational autoencoder (VAE)."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder (VAE).

    Args:
        input_dims (int): The dimensionality of the input data.
        hidden_layers (list): A list of integers where each value specifies
                              number of nodes in a hidden layer for the encoder
                              The decoder will mirror this structure.
        latent_dims (int): dimensionality of the latent space representation.

    Returns:
        tuple: A tuple containing:
            - encoder (keras.Model): The encoder model that outputs:
                - The latent space representation (z),
                - The mean of the latent distribution (z_mean),
                - The log variance of the latent distribution (z_log_var).
            - decoder (keras.Model): The decoder model that reconstructs
              the input
              from the latent space representation.
            - auto (keras.Model): The full autoencoder model that combines
              the encoder and decoder.

    Notes:
        - The encoder and decoder both use `relu` activations for all layers
          except the final layers, which use no activation for the mean
          and log variance
          in the encoder, and `sigmoid` for the output of the decoder.
        - The KL divergence is included as a loss term in the autoencoder
          to ensure a structured latent space.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,), name="encoder_input")
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(
            units, activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)

    z_mean = keras.layers.Dense(
        latent_dims, activation=None, name="z_mean")(x)
    z_log_var = keras.layers.Dense(
        latent_dims, activation=None, name="z_log_var")(x)

    def sampling(args):
        """
        Samples from the latent space using the reparameterization trick.

        Args:
            args (tuple): A tuple containing the mean and log variance of the
                          latent distribution.

        Returns:
            keras.Tensor: A sampled latent vector.
        """
        mean, log_var = args
        epsilon = (
            keras.backend.random_normal(
                shape=keras.backend.shape(mean), mean=0., stddev=1.0))
        return mean + keras.backend.exp(0.5 * log_var) * epsilon

    z = keras.layers.Lambda(sampling, name="z")([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,), name="decoder_input")
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(
            units, activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = keras.layers.BatchNormalization()(x)

    outputs = keras.layers.Dense(
        input_dims, activation='sigmoid', name="decoder_output")(x)
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # Autoencoder
    reconstructed = decoder(z)
    auto = keras.Model(inputs, reconstructed, name="autoencoder")

    # KL Divergence Loss
    kl_loss = -0.5 * keras.backend.mean(
        1 + z_log_var - keras.backend.square(z_mean)
        - keras.backend.exp(z_log_var)
    )
    auto.add_loss(1e-4 * kl_loss)  # Add KL loss to the model

    # Compile the autoencoder with binary cross-entropy loss
    auto.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy")

    return encoder, decoder, auto
