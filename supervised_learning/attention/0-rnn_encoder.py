#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to encode for machine translation
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Class that implements an RNN encoder for machine translation.

    This class defines the initializer for the RNN encoder,
    including the embedding layer,
    GRU layer, and methods to initialize the hidden state and
    process the input sequence.

    Attributes:
        batch: The batch size.
        units: The number of hidden units in the RNN cell.
        embedding: A tf.keras.layers.Embedding layer that converts words
        from the vocabulary into an embedding vector.
        gru: A tf.keras.layers.GRU layer with 'units' units.
    """

    def __init__(self, vocab, embedding, units, batch):
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Initializes the hidden state of the RNN encoder to a tensor of zeros.

        Returns:
            A tensor of shape (batch, units) containing the initialized
            hidden states.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Encodes a sequence of input tokens using the RNN encoder.

        Args:
            x: A tensor of shape (batch, input_seq_len) containing the input
            sequence as word indices within the vocabulary.
            initial_state tensor of shape (batch, units) containing the initial
            hidden state of the RNN encoder.

        Returns:
            outputs: A tensor of shape (batch, input_seq_len, units) containing
            the outputs of the RNN encoder.
            hidden: A tensor of shape (batch, units) containing the last hidden
            state of the RNN encoder.
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
