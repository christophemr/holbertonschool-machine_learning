#!/usr/bin/env python3
"""
Defines function that creates and trains a gensim word2vec model
"""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Word2Vec model.

    Parameters:
        sentences (list): List of sentences to be trained on
        vector_size (int): Dimensionality of the embedding layer
        min_count (int): Minimum number of occurrences of a word
        for use in training
        window (int): Maximum distance between the current and
        predicted word within a sentence
        negative (int): Size of negative sampling
        cbow (bool): Training type; True for CBOW, False for Skip-gram
        epochs (int): Number of iterations to train over
        seed (int): Seed for the random number generator
        workers (int): Number of worker threads to train the model

    Returns:
        model: The trained Word2Vec model
    """
    cbow_or_sg = 0 if cbow else 1

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=cbow_or_sg,
        negative=negative,
        seed=seed,
        workers=workers,
    )

    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
