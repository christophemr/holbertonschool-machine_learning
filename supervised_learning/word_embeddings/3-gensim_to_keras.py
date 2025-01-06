#!/usr/bin/env python3
"""
Defines function that converts gensim word2vec model to Keras Embedding layer
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model to a Keras Embedding layer.

    Parameters:
        model: A trained gensim Word2Vec model.

    Returns:
        tf.Keras.layer.Embedding: embedding layer
    """
    # Extract weights from the gensim model
    weights = model.wv.vectors

    vocab_size, embedding_dim = weights.shape

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(weights),
        trainable=True
    )

    return embedding_layer
