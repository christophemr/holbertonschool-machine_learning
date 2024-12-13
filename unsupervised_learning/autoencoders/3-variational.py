#!/usr/bin/env python3
"""Creates a variational autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates an autoencoder:

    input_dims: An integer containing the dimensions of the model input.
    hidden_layers: A list containing the number of nodes for each hidden
        layer in the encoder, respectively.
    latent_dims: An integer containing the dimensions of the latent space
        representation.

    Returns: (encoder, decoder, auto)
        encoder: The encoder model.
        decoder: The decoder model.
        auto: The full autoencoder model.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)

    # Latent space layers
    z_mean = keras.layers.Dense(latent_dims, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dims, name="z_log_var")(x)

    # Sampling function (reparameterization trick)
    def sampling(args):
        mean, log_var = args
        epsilon = (
            keras.backend.random_normal(
                shape=keras.backend.shape(mean), mean=0., stddev=1.0))
        return mean + keras.backend.exp(log_var * 0.5) * epsilon

    z = keras.layers.Lambda(sampling, name="z")([z_mean, z_log_var])

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)

    # Output layer with sigmoid activation
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # Define the full autoencoder model
    reconstructed = decoder(z)

    # KL Divergence Loss Calculation
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_var - keras.backend.square(z_mean)
        - keras.backend.exp(z_log_var),
        axis=-1
    )

    # Add KL divergence as a loss term
    auto = keras.Model(inputs=inputs, outputs=reconstructed)
    auto.add_loss(1e-3 * keras.backend.mean(kl_loss))  # Increased KL weight

    # Compile with reconstruction loss
    auto.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy")

    # Encoder outputs both the latent representation and the latent parameters
    encoder = keras.Model(inputs=inputs, outputs=[z, z_mean, z_log_var])

    return encoder, decoder, auto
