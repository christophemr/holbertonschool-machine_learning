#!/usr/bin/env python3
"""
Defines function that creates all masks for training/validation
"""

import tensorflow as tf


def padding_mask(seq):
    """
    Creates a padding mask for a sequence.

    Args:
        seq (tf.Tensor): Tensor of shape (batch_size, seq_len)
        containing the sequence.

    Returns:
        tf.Tensor: Padding mask of shape (batch_size, 1, 1, seq_len).
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def lookahead_mask(size):
    """
    Creates a lookahead mask to prevent attending to future tokens.

    Args:
        size (int): Size of the sequence.

    Returns:
        tf.Tensor: Lookahead mask of shape (size, size).
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """
    Creates all masks for training/validation.

    Args:
        inputs (tf.Tensor): Tensor of shape (batch_size, seq_len_in)
        containing the input sentence.
        target (tf.Tensor): Tensor of shape (batch_size, seq_len_out)
        containing the target sentence.

    Returns:
        tuple: encoder_mask, combined_mask, decoder_mask
            - encoder_mask: Padding mask for the encoder
            (batch_size, 1, 1, seq_len_in).
            - combined_mask: Combined mask for the decoder's first attention
            block (batch_size, 1, seq_len_out, seq_len_out).
            - decoder_mask: Padding mask for the encoder-decoder attention
            (batch_size, 1, 1, seq_len_in).
    """
    # Encoder padding mask
    encoder_mask = padding_mask(inputs)

    # Decoder target padding mask
    decoder_padding_mask = padding_mask(target)

    # Lookahead mask
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = lookahead_mask(seq_len_out)

    # Combine lookahead mask and padding mask
    combined_mask = tf.maximum(
        look_ahead_mask, decoder_padding_mask)

    # Decoder mask on encoder's output
    decoder_mask = encoder_mask

    return encoder_mask, combined_mask, decoder_mask
