#!/usr/bin/env python3
"""
Defines function that converts gensim word2vec model to Keras Embedding layer
"""

from keras.layers import Embedding


def gensim_to_keras(model):
    """
    Converts a gensim Word2Vec model to a Keras Embedding layer.

    Parameters:
        model: A trained gensim Word2Vec model.

    Returns:
        Keras Embedding layer initialized with Word2Vec weights.
    """
    # Extract weights from the gensim model
    weights = model.wv.vectors

    vocab_size, embedding_dim = weights.shape

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )

    return embedding_layer
