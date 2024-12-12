#!/usr/bin/env python3
"""creates a convolutional autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.

    Args:
        input_dims (tuple): Dimensions of the model input.
        filters (list): List of integers, the number of filters for each
        convolutional layer in the encoder.
        latent_dims (tuple): Dimensions of the latent space representation.

    Returns:
        encoder (Model): The encoder model.
        decoder (Model): The decoder model.
        auto (Model): The full autoencoder model.
    """
    # Encoder
    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = (keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                 padding='same', activation='relu')(x))
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Latent space
    latent = (keras.layers.Conv2D(filters=filters[-1], kernel_size=(3, 3),
                                  padding='same', activation='relu')(x))

    encoder = keras.Model(encoder_input, latent, name="encoder")

    # Decoder
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    for f in filters[::-1]:
        x = (keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                 padding='same', activation='relu')(x))
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Final convolution to match input dimensions exactly
    x = (keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                             padding='same', activation='sigmoid')(x))

    # Ensure decoder output matches input dimensions
    x = (keras.layers.Lambda(
        lambda t: t[:, :input_dims[0], :input_dims[1], :])(x))

    decoder = keras.Model(decoder_input, x, name="decoder")

    # Autoencoder
    autoencoder_input = encoder_input
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)

    auto = keras.Model(autoencoder_input, decoded, name="autoencoder")
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
